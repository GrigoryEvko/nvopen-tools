// Function: sub_2BE4CE0
// Address: 0x2be4ce0
//
__int64 __fastcall sub_2BE4CE0(char *a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rax
  char v4; // al
  __int64 v5; // r12
  unsigned int v6; // r13d
  char *v7; // r14
  volatile signed __int32 **v8; // rsi
  __int64 v9; // r12
  char v10; // al
  unsigned int v11; // r12d
  unsigned int v13; // eax
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // r14
  char *v19; // [rsp+0h] [rbp-60h]
  char v20; // [rsp+Dh] [rbp-53h]
  char v21; // [rsp+Eh] [rbp-52h]
  char v22; // [rsp+Fh] [rbp-51h]
  volatile signed __int32 *v23[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF

  v2 = a1[8];
  v3 = sub_222F790(*(_QWORD **)(*(_QWORD *)a1 + 104LL), a2);
  v4 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v3 + 32LL))(v3, v2);
  v5 = *(_QWORD *)a1;
  LOBYTE(v23[0]) = v4;
  if ( !sub_2BE37E0(*(char **)v5, *(_BYTE **)(v5 + 8), (char *)v23) )
  {
    v6 = a1[8];
    v19 = *(char **)(v5 + 56);
    if ( v19 == *(char **)(v5 + 48) )
    {
LABEL_12:
      LOBYTE(v13) = sub_2BDBFE0(*(_QWORD **)(v5 + 112), v6, *(_WORD *)(v5 + 96), *(_BYTE *)(v5 + 98));
      v11 = v13;
      if ( !(_BYTE)v13 )
      {
        v14 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
        sub_2BE40B0((__int64)v23, *(_QWORD **)(*(_QWORD *)a1 + 112LL), a1 + 8, (__int64)(a1 + 9));
        v15 = sub_2BDD0F0(*(_QWORD *)(*(_QWORD *)a1 + 24LL), *(_QWORD *)(*(_QWORD *)a1 + 32LL), (__int64)v23);
        if ( (__int64 *)v23[0] != &v24 )
          j_j___libc_free_0((unsigned __int64)v23[0]);
        if ( v15 == v14 )
        {
          v16 = *(_QWORD *)a1;
          v17 = *(_QWORD *)(*(_QWORD *)a1 + 72LL);
          v18 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
          if ( v17 == v18 )
            return v11;
          while ( sub_2BDBFE0(*(_QWORD **)(v16 + 112), (unsigned int)a1[8], *(_WORD *)v17, *(_BYTE *)(v17 + 2)) )
          {
            v17 += 4;
            if ( v18 == v17 )
              return v11;
            v16 = *(_QWORD *)a1;
          }
        }
      }
    }
    else
    {
      v7 = *(char **)(v5 + 48);
      while ( 1 )
      {
        v8 = *(volatile signed __int32 ***)(v5 + 104);
        v20 = *v7;
        v22 = v7[1];
        sub_2208E20(v23, v8);
        v9 = sub_222F790(v23, (__int64)v8);
        sub_2209150(v23);
        v21 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 32LL))(v9, v6);
        v10 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 16LL))(v9, v6);
        if ( v20 <= v21 && v22 >= v21 )
          break;
        if ( v20 <= v10 && v22 >= v10 )
          break;
        v5 = *(_QWORD *)a1;
        v7 += 2;
        if ( v19 == v7 )
        {
          v6 = a1[8];
          goto LABEL_12;
        }
      }
    }
  }
  return 1;
}
