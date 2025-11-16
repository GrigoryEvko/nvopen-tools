// Function: sub_1A56F40
// Address: 0x1a56f40
//
_QWORD *__fastcall sub_1A56F40(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // rsi
  __int64 v5; // r15
  _QWORD *v6; // r13
  char v7; // al
  const char *v8; // rbx
  _QWORD *v9; // rax
  unsigned int v11; // esi
  int v12; // eax
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // [rsp+18h] [rbp-98h] BYREF
  void *v18; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v19[2]; // [rsp+28h] [rbp-88h] BYREF
  __int64 v20; // [rsp+38h] [rbp-78h]
  __int64 v21; // [rsp+40h] [rbp-70h]
  const char *v22; // [rsp+50h] [rbp-60h] BYREF
  __int64 v23; // [rsp+58h] [rbp-58h] BYREF
  __int64 v24; // [rsp+60h] [rbp-50h]
  __int64 v25; // [rsp+68h] [rbp-48h]
  __int64 v26; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 56);
  v4 = *(_QWORD *)a1;
  v22 = ".us";
  LOWORD(v24) = 259;
  v17 = (_QWORD *)sub_1AB5760(a2, v4, &v22, v3, 0, 0);
  sub_1580B80(v17, **(_QWORD **)(a1 + 8));
  sub_15CDD90(*(_QWORD *)(a1 + 16), &v17);
  v5 = *(_QWORD *)a1;
  v20 = a2;
  v19[1] = 0;
  v6 = v17;
  v19[0] = 2;
  if ( a2 != -16 && a2 != -8 )
    sub_164C220((__int64)v19);
  v21 = v5;
  v18 = &unk_49E6B50;
  v7 = sub_12E4800(v5, (__int64)&v18, &v22);
  v8 = v22;
  if ( !v7 )
  {
    v11 = *(_DWORD *)(v5 + 24);
    v12 = *(_DWORD *)(v5 + 16);
    ++*(_QWORD *)v5;
    v13 = v12 + 1;
    if ( 4 * v13 >= 3 * v11 )
    {
      v11 *= 2;
    }
    else if ( v11 - *(_DWORD *)(v5 + 20) - v13 > v11 >> 3 )
    {
LABEL_15:
      *(_DWORD *)(v5 + 16) = v13;
      v23 = 2;
      v24 = 0;
      v25 = -8;
      v26 = 0;
      if ( *((_QWORD *)v8 + 3) != -8 )
        --*(_DWORD *)(v5 + 20);
      v22 = (const char *)&unk_49EE2B0;
      sub_1455FA0((__int64)&v23);
      v14 = *((_QWORD *)v8 + 3);
      v15 = v20;
      if ( v14 != v20 )
      {
        if ( v14 != -8 && v14 != 0 && v14 != -16 )
        {
          sub_1649B30((_QWORD *)v8 + 1);
          v15 = v20;
        }
        *((_QWORD *)v8 + 3) = v15;
        if ( v15 != -8 && v15 != 0 && v15 != -16 )
          sub_1649AC0((unsigned __int64 *)v8 + 1, v19[0] & 0xFFFFFFFFFFFFFFF8LL);
      }
      v16 = v21;
      *((_QWORD *)v8 + 5) = 6;
      *((_QWORD *)v8 + 6) = 0;
      *((_QWORD *)v8 + 4) = v16;
      *((_QWORD *)v8 + 7) = 0;
      goto LABEL_5;
    }
    sub_12E48B0(v5, v11);
    sub_12E4800(v5, (__int64)&v18, &v22);
    v8 = v22;
    v13 = *(_DWORD *)(v5 + 16) + 1;
    goto LABEL_15;
  }
LABEL_5:
  v18 = &unk_49EE2B0;
  sub_1455FA0((__int64)v19);
  v9 = (_QWORD *)*((_QWORD *)v8 + 7);
  if ( v6 != v9 )
  {
    if ( v9 != 0 && v9 + 1 != 0 && v9 != (_QWORD *)-16LL )
      sub_1649B30((_QWORD *)v8 + 5);
    *((_QWORD *)v8 + 7) = v6;
    if ( v6 + 1 != 0 && v6 != 0 && v6 != (_QWORD *)-16LL )
      sub_164C220((__int64)(v8 + 40));
  }
  return v17;
}
