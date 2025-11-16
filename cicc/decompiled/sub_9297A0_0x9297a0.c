// Function: sub_9297A0
// Address: 0x9297a0
//
__int64 __fastcall sub_9297A0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v3; // r12
  __int64 v5; // r13
  __int64 v6; // rdi
  unsigned int **v7; // r15
  unsigned int v8; // eax
  unsigned int **v9; // r15
  unsigned int *v10; // rdi
  __int64 (__fastcall *v11)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  __int64 v12; // r13
  unsigned int *v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rsi
  unsigned int v18; // [rsp+Ch] [rbp-94h]
  char *v19; // [rsp+10h] [rbp-90h] BYREF
  char v20; // [rsp+30h] [rbp-70h]
  char v21; // [rsp+31h] [rbp-6Fh]
  _QWORD v22[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v23; // [rsp+60h] [rbp-40h]

  v3 = (_BYTE *)a3;
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_QWORD *)(a3 + 8);
  if ( v5 != v6 )
  {
    v7 = *(unsigned int ***)(a1 + 8);
    v22[0] = "sh_prom";
    v23 = 259;
    v18 = sub_BCB060(v6);
    v8 = sub_BCB060(v5);
    v3 = (_BYTE *)sub_929600(v7, (unsigned int)(v18 <= v8) + 38, (__int64)v3, v5, (__int64)v22, 0, (unsigned int)v19, 0);
  }
  v9 = *(unsigned int ***)(a1 + 8);
  v21 = 1;
  v19 = "shl";
  v20 = 3;
  v10 = v9[10];
  v11 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v10 + 32LL);
  if ( v11 != sub_9201A0 )
  {
    v12 = v11((__int64)v10, 25u, (_BYTE *)a2, v3, 0, 0);
    goto LABEL_8;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v3 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(25) )
      v12 = sub_AD5570(25, a2, v3, 0, 0);
    else
      v12 = sub_AABE40(25, a2, v3);
LABEL_8:
    if ( v12 )
      return v12;
  }
  v23 = 257;
  v12 = sub_B504D0(25, a2, v3, v22, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, char **, unsigned int *, unsigned int *))(*(_QWORD *)v9[11] + 16LL))(
    v9[11],
    v12,
    &v19,
    v9[7],
    v9[8]);
  v14 = *v9;
  v15 = (__int64)&(*v9)[4 * *((unsigned int *)v9 + 2)];
  if ( *v9 != (unsigned int *)v15 )
  {
    do
    {
      v16 = *((_QWORD *)v14 + 1);
      v17 = *v14;
      v14 += 4;
      sub_B99FD0(v12, v17, v16);
    }
    while ( (unsigned int *)v15 != v14 );
  }
  return v12;
}
