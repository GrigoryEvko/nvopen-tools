// Function: sub_12A1C70
// Address: 0x12a1c70
//
__int64 __fastcall sub_12A1C70(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  char v4; // al
  __int64 v5; // r14
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rdx
  bool v9; // zf
  int v10; // r15d
  int v11; // eax
  int v12; // r9d
  __int64 v13; // r12
  __int64 v15; // [rsp+8h] [rbp-78h]
  int v16; // [rsp+14h] [rbp-6Ch]
  __int64 v17; // [rsp+18h] [rbp-68h]
  _BYTE *v18; // [rsp+20h] [rbp-60h] BYREF
  __int64 v19; // [rsp+28h] [rbp-58h]
  _BYTE v20[80]; // [rsp+30h] [rbp-50h] BYREF

  v2 = a2;
  v19 = 0x400000000LL;
  v3 = *(_QWORD *)(a2 + 128);
  v18 = v20;
  v17 = 8 * v3;
  v4 = *(_BYTE *)(a2 + 140);
  if ( *(char *)(a2 + 142) < 0 || v4 != 12 )
  {
    v5 = a1 + 16;
    v16 = 8 * *(_DWORD *)(a2 + 136);
    if ( v4 == 8 )
      goto LABEL_4;
LABEL_11:
    v8 = (unsigned int)v19;
    goto LABEL_7;
  }
  v5 = a1 + 16;
  v16 = 8 * sub_8D4AB0(a2);
  if ( *(_BYTE *)(a2 + 140) != 8 )
    goto LABEL_11;
  do
  {
LABEL_4:
    v6 = sub_15A6850(v5, 0, *(_QWORD *)(v2 + 176));
    v7 = (unsigned int)v19;
    if ( (unsigned int)v19 >= HIDWORD(v19) )
    {
      v15 = v6;
      sub_16CD150(&v18, v20, 0, 8);
      v7 = (unsigned int)v19;
      v6 = v15;
    }
    *(_QWORD *)&v18[8 * v7] = v6;
    v2 = *(_QWORD *)(v2 + 160);
    v8 = (unsigned int)(v19 + 1);
    v9 = *(_BYTE *)(v2 + 140) == 8;
    LODWORD(v19) = v19 + 1;
  }
  while ( v9 );
LABEL_7:
  v10 = sub_15A5DC0(v5, v18, v8);
  v11 = sub_12A0C10(a1, v2);
  v13 = sub_15A6E90(v5, v17, v16, v11, v10, v12, (__int64)byte_3F871B3, 0);
  if ( v18 != v20 )
    _libc_free(v18, v17);
  return v13;
}
