// Function: sub_929960
// Address: 0x929960
//
__int64 __fastcall sub_929960(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r13
  __int64 v7; // r15
  __int64 v8; // rdi
  unsigned int **v9; // r10
  unsigned int v10; // eax
  char v11; // al
  unsigned int **v12; // r15
  bool v13; // zf
  unsigned int *v14; // rdi
  __int64 (__fastcall *v15)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8); // rax
  __int64 v16; // r12
  unsigned int *v18; // rbx
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned int *v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rsi
  unsigned int **v26; // [rsp+0h] [rbp-A0h]
  unsigned int v27; // [rsp+Ch] [rbp-94h]
  unsigned int v28[8]; // [rsp+10h] [rbp-90h] BYREF
  char v29; // [rsp+30h] [rbp-70h]
  char v30; // [rsp+31h] [rbp-6Fh]
  _QWORD v31[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v32; // [rsp+60h] [rbp-40h]

  v4 = (_BYTE *)a3;
  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(_QWORD *)(a3 + 8);
  if ( v7 != v8 )
  {
    v9 = *(unsigned int ***)(a1 + 8);
    v31[0] = "sh_prom";
    v26 = v9;
    v32 = 259;
    v27 = sub_BCB060(v8);
    v10 = sub_BCB060(v7);
    v4 = (_BYTE *)sub_929600(v26, (unsigned int)(v27 <= v10) + 38, (__int64)v4, v7, (__int64)v31, 0, v28[0], 0);
  }
  v11 = sub_91B730(a4);
  v12 = *(unsigned int ***)(a1 + 8);
  v30 = 1;
  v13 = v11 == 0;
  v29 = 3;
  *(_QWORD *)v28 = "shr";
  v14 = v12[10];
  v15 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8))(*(_QWORD *)v14 + 24LL);
  if ( !v13 )
  {
    if ( v15 == sub_920250 )
    {
      if ( *(_BYTE *)a2 > 0x15u || *v4 > 0x15u )
      {
LABEL_20:
        v32 = 257;
        v16 = sub_B504D0(26, a2, v4, v31, 0, 0);
        (*(void (__fastcall **)(unsigned int *, __int64, unsigned int *, unsigned int *, unsigned int *))(*(_QWORD *)v12[11] + 16LL))(
          v12[11],
          v16,
          v28,
          v12[7],
          v12[8]);
        v22 = *v12;
        v23 = (__int64)&(*v12)[4 * *((unsigned int *)v12 + 2)];
        if ( *v12 != (unsigned int *)v23 )
        {
          do
          {
            v24 = *((_QWORD *)v22 + 1);
            v25 = *v22;
            v22 += 4;
            sub_B99FD0(v16, v25, v24);
          }
          while ( (unsigned int *)v23 != v22 );
        }
        return v16;
      }
      if ( (unsigned __int8)sub_AC47B0(26) )
        v16 = sub_AD5570(26, a2, v4, 0, 0);
      else
        v16 = sub_AABE40(26, a2, v4);
    }
    else
    {
      v16 = v15((__int64)v14, 26u, (_BYTE *)a2, v4, 0);
    }
    if ( v16 )
      return v16;
    goto LABEL_20;
  }
  if ( v15 != sub_920250 )
  {
    v16 = v15((__int64)v14, 27u, (_BYTE *)a2, v4, 0);
    goto LABEL_16;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v4 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(27) )
      v16 = sub_AD5570(27, a2, v4, 0, 0);
    else
      v16 = sub_AABE40(27, a2, v4);
LABEL_16:
    if ( v16 )
      return v16;
  }
  v32 = 257;
  v16 = sub_B504D0(27, a2, v4, v31, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, unsigned int *, unsigned int *, unsigned int *))(*(_QWORD *)v12[11]
                                                                                                  + 16LL))(
    v12[11],
    v16,
    v28,
    v12[7],
    v12[8]);
  v18 = *v12;
  v19 = (__int64)&(*v12)[4 * *((unsigned int *)v12 + 2)];
  if ( *v12 != (unsigned int *)v19 )
  {
    do
    {
      v20 = *((_QWORD *)v18 + 1);
      v21 = *v18;
      v18 += 4;
      sub_B99FD0(v16, v21, v20);
    }
    while ( (unsigned int *)v19 != v18 );
  }
  return v16;
}
