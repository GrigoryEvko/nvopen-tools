// Function: sub_393B7F0
// Address: 0x393b7f0
//
__int64 *__fastcall sub_393B7F0(__int64 *a1, __int64 a2, int *a3, size_t a4, unsigned __int64 *a5)
{
  __int64 *v6; // r13
  _QWORD *v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rax
  _WORD *v10; // rdx
  _QWORD *v11; // rax
  int v12; // r11d
  __int64 *v13; // r8
  __int64 v14; // r9
  int *v15; // rsi
  int v16; // ecx
  __int64 v17; // r13
  size_t v18; // rdx
  __int64 v19; // r12
  unsigned __int64 v20; // r15
  unsigned __int64 v21; // rbx
  int v22; // eax
  unsigned __int64 v23; // rdx
  __int64 *v25; // [rsp+8h] [rbp-A8h]
  __int64 v26; // [rsp+10h] [rbp-A0h]
  size_t v27; // [rsp+18h] [rbp-98h]
  int s2a; // [rsp+28h] [rbp-88h]
  int v30; // [rsp+30h] [rbp-80h]
  int v32; // [rsp+4Ch] [rbp-64h] BYREF
  _QWORD v33[12]; // [rsp+50h] [rbp-60h] BYREF

  v6 = a1;
  v7 = *(_QWORD **)(a2 + 8);
  v8 = sub_3938FB0((__int64)(v7 + 4), a3, a4);
  v9 = *(_QWORD *)(v7[2] + 8 * (v8 & (*v7 - 1LL)));
  if ( v9 && (v10 = (_WORD *)(v7[3] + v9), v11 = v10 + 1, v12 = (unsigned __int16)*v10, *v10) )
  {
    v13 = a1;
    v14 = (__int64)(v7 + 4);
    v15 = a3;
    v16 = 0;
    v17 = v8;
    v18 = a4;
    while ( 1 )
    {
      v19 = v11[1];
      v20 = v11[2];
      v21 = (unsigned __int64)(v11 + 3);
      if ( v17 == *v11 && v18 == v19 )
      {
        s2a = v16;
        v30 = v12;
        if ( !v18 )
          break;
        v25 = v13;
        v26 = v14;
        v27 = v18;
        v22 = memcmp(v11 + 3, v15, v18);
        v12 = v30;
        v18 = v27;
        v14 = v26;
        v13 = v25;
        v16 = s2a;
        if ( !v22 )
          break;
      }
      ++v16;
      v11 = (_QWORD *)(v21 + v19 + v20);
      if ( v12 == v16 )
      {
        v6 = v13;
        goto LABEL_12;
      }
    }
    v6 = v13;
    v33[0] = v21;
    v33[1] = v19;
    *a5 = sub_393B300(v14, v21, v19, (unsigned int *)(v21 + v19), v20);
    a5[1] = v23;
    if ( v23 )
    {
      *v6 = 1;
    }
    else
    {
      v32 = 9;
      sub_3939440(v6, &v32);
    }
  }
  else
  {
LABEL_12:
    LODWORD(v33[0]) = 10;
    sub_3939440(v6, (int *)v33);
  }
  return v6;
}
