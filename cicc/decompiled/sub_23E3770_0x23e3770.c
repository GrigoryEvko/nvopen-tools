// Function: sub_23E3770
// Address: 0x23e3770
//
void __fastcall sub_23E3770(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 *v7; // rdi
  _QWORD *v8; // rax
  __int64 v9; // r13
  unsigned __int64 v10; // rsi
  __int64 v11; // rax
  int v12; // ecx
  _QWORD *v13; // rdx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // [rsp-10h] [rbp-40h]
  __int64 v17; // [rsp-8h] [rbp-38h]
  _QWORD v18[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_BD5C60(a2);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 72) = v3;
  *(_QWORD *)(a1 + 80) = a1 + 128;
  *(_QWORD *)(a1 + 88) = a1 + 136;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  *(_WORD *)(a1 + 108) = 512;
  *(_QWORD *)(a1 + 128) = &unk_49DA100;
  *(_WORD *)(a1 + 64) = 0;
  *(_BYTE *)(a1 + 110) = 7;
  *(_QWORD *)(a1 + 136) = &unk_49DA0B0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  sub_D5F1F0(a1, a2);
  v4 = sub_B43CB0(a2);
  sub_B33910(v18, (__int64 *)a1);
  v5 = v18[0];
  if ( v18[0] )
    goto LABEL_2;
  v6 = sub_B92180(v4);
  if ( v6 )
  {
    v7 = (__int64 *)(*(_QWORD *)(v6 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(v6 + 8) & 4) != 0 )
      v7 = (__int64 *)*v7;
    v8 = sub_B01860(v7, 0, 0, v6, 0, 0, 0, 1);
    sub_B10CB0(v18, (__int64)v8);
    v9 = v18[0];
    if ( v18[0] )
    {
      v10 = *(unsigned int *)(a1 + 8);
      v11 = *(_QWORD *)a1;
      v12 = *(_DWORD *)(a1 + 8);
      v13 = (_QWORD *)(*(_QWORD *)a1 + 16 * v10);
      if ( *(_QWORD **)a1 != v13 )
      {
        while ( *(_DWORD *)v11 )
        {
          v11 += 16;
          if ( v13 == (_QWORD *)v11 )
            goto LABEL_14;
        }
        *(_QWORD *)(v11 + 8) = v18[0];
        goto LABEL_13;
      }
LABEL_14:
      v14 = *(unsigned int *)(a1 + 12);
      if ( v10 >= v14 )
      {
        v15 = v10 + 1;
        if ( v14 < v15 )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v15, 0x10u, v16, v17);
          v13 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
        }
        *v13 = 0;
        v13[1] = v9;
        v9 = v18[0];
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        if ( v13 )
        {
          *(_DWORD *)v13 = 0;
          v13[1] = v9;
          v12 = *(_DWORD *)(a1 + 8);
          v9 = v18[0];
        }
        *(_DWORD *)(a1 + 8) = v12 + 1;
      }
    }
    else
    {
      sub_93FB40(a1, 0);
      v9 = v18[0];
    }
    if ( v9 )
    {
LABEL_13:
      v5 = v9;
LABEL_2:
      sub_B91220((__int64)v18, v5);
    }
  }
}
