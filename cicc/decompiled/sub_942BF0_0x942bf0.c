// Function: sub_942BF0
// Address: 0x942bf0
//
__int64 __fastcall sub_942BF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  char v4; // al
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  bool v9; // zf
  int v10; // r15d
  int v11; // eax
  __int64 v12; // r12
  __int64 v14; // [rsp+8h] [rbp-78h]
  int v15; // [rsp+14h] [rbp-6Ch]
  __int64 v16; // [rsp+18h] [rbp-68h]
  _BYTE *v17; // [rsp+20h] [rbp-60h] BYREF
  __int64 v18; // [rsp+28h] [rbp-58h]
  _BYTE v19[80]; // [rsp+30h] [rbp-50h] BYREF

  v2 = a2;
  v18 = 0x400000000LL;
  v3 = *(_QWORD *)(a2 + 128);
  v17 = v19;
  v16 = 8 * v3;
  v4 = *(_BYTE *)(a2 + 140);
  if ( *(char *)(a2 + 142) < 0 || v4 != 12 )
  {
    v5 = a1 + 16;
    v15 = 8 * *(_DWORD *)(a2 + 136);
    if ( v4 == 8 )
      goto LABEL_4;
LABEL_11:
    v8 = (unsigned int)v18;
    goto LABEL_7;
  }
  v5 = a1 + 16;
  v15 = 8 * sub_8D4AB0(a2);
  if ( *(_BYTE *)(a2 + 140) != 8 )
    goto LABEL_11;
  do
  {
LABEL_4:
    v6 = sub_ADD550(v5, 0, *(_QWORD *)(v2 + 176));
    v7 = (unsigned int)v18;
    if ( (unsigned __int64)(unsigned int)v18 + 1 > HIDWORD(v18) )
    {
      v14 = v6;
      sub_C8D5F0(&v17, v19, (unsigned int)v18 + 1LL, 8);
      v7 = (unsigned int)v18;
      v6 = v14;
    }
    *(_QWORD *)&v17[8 * v7] = v6;
    v2 = *(_QWORD *)(v2 + 160);
    v8 = (unsigned int)(v18 + 1);
    v9 = *(_BYTE *)(v2 + 140) == 8;
    LODWORD(v18) = v18 + 1;
  }
  while ( v9 );
LABEL_7:
  v10 = sub_ADCD70(v5, v17, v8);
  v11 = sub_941B90(a1, v2);
  v12 = sub_ADE2A0(v5, v16, v15, v11, v10, 0, 0, 0, 0);
  if ( v17 != v19 )
    _libc_free(v17, v16);
  return v12;
}
