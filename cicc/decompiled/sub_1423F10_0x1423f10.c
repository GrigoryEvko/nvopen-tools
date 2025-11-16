// Function: sub_1423F10
// Address: 0x1423f10
//
__int64 __fastcall sub_1423F10(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned int v3; // r8d
  __int64 v5; // rax
  bool v6; // zf
  __int64 v7; // rcx
  _QWORD *v8; // rdx
  unsigned __int64 v9; // r12
  __int64 *i; // r14
  __int64 v11; // rax
  int v12; // r12d
  __int64 v13; // r13
  unsigned int j; // r12d
  _QWORD *v15; // r15
  char v16; // al
  unsigned int v17; // r12d
  int v18; // [rsp+4h] [rbp-BCh]
  __int64 v19; // [rsp+8h] [rbp-B8h]
  int v21; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v22; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v23; // [rsp+28h] [rbp-98h] BYREF
  _QWORD v24[6]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v25[12]; // [rsp+60h] [rbp-60h] BYREF

  v21 = *(_DWORD *)(a1 + 24);
  if ( v21 )
  {
    v5 = *(_QWORD *)(a1 + 8);
    v6 = *(_BYTE *)a2 == 0;
    v24[0] = 0;
    v24[1] = -8;
    v7 = *(_QWORD *)(a2 + 8);
    v19 = v5;
    memset(&v24[2], 0, 32);
    v25[0] = 0;
    v25[1] = -16;
    memset(&v25[2], 0, 32);
    if ( v6 )
    {
      LODWORD(v23) = ((unsigned int)v7 >> 9)
                   ^ ((unsigned int)v7 >> 4)
                   ^ ((unsigned int)*(_QWORD *)(a2 + 40) >> 9)
                   ^ ((unsigned int)*(_QWORD *)(a2 + 40) >> 4)
                   ^ ((unsigned int)*(_QWORD *)(a2 + 32) >> 9)
                   ^ ((unsigned int)*(_QWORD *)(a2 + 32) >> 4)
                   ^ (37 * *(_QWORD *)(a2 + 16))
                   ^ ((unsigned int)*(_QWORD *)(a2 + 24) >> 4)
                   ^ ((unsigned int)*(_QWORD *)(a2 + 24) >> 9);
      v12 = sub_1423D60((_BYTE *)a2, &v23);
    }
    else
    {
      v8 = (_QWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) - 72);
      if ( (v7 & 4) != 0 )
        v8 = (_QWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) - 24);
      LODWORD(v23) = ((unsigned int)*v8 >> 4) ^ ((unsigned int)*v8 >> 9);
      v22 = sub_1423D60((_BYTE *)a2, &v23);
      v23 = *(_QWORD *)(a2 + 8);
      v9 = sub_134EF80(&v23);
      for ( i = (__int64 *)((v23 & 0xFFFFFFFFFFFFFFF8LL)
                          - 24LL * (*(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
            (__int64 *)v9 != i;
            v22 = sub_1423E30(&v22, &v23) )
      {
        v11 = *i;
        i += 3;
        LODWORD(v23) = ((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9);
      }
      v12 = v22;
    }
    v13 = 0;
    v18 = 1;
    for ( j = (v21 - 1) & v12; ; j = (v21 - 1) & v17 )
    {
      v15 = (_QWORD *)(v19 + 96LL * j);
      v3 = sub_1423BB0((_QWORD *)a2, (__int64)v15);
      if ( (_BYTE)v3 )
      {
        *a3 = v15;
        return v3;
      }
      if ( (unsigned __int8)sub_1423BB0(v15, (__int64)v24) )
        break;
      v16 = sub_1423BB0(v15, (__int64)v25);
      if ( !v13 && v16 )
        v13 = v19 + 96LL * j;
      v17 = v18 + j;
      ++v18;
    }
    v3 = 0;
    if ( !v13 )
      v13 = v19 + 96LL * j;
    *a3 = v13;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
  return v3;
}
