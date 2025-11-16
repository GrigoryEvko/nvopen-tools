// Function: sub_2AF5B40
// Address: 0x2af5b40
//
__int64 __fastcall sub_2AF5B40(__int64 a1)
{
  __int64 *v2; // r12
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rsi
  int v6; // r12d
  char **v7; // r13
  char **i; // r14
  char *v9; // rdi
  int v10; // eax
  __int64 *v11; // r14
  __int64 *v12; // r13
  __int64 v13; // rdi
  int v14; // eax
  _BYTE *v15; // rdi
  int v16; // r14d
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // r15
  __int64 v20; // rdx
  unsigned int v21; // ebx
  _BYTE *v23; // [rsp+20h] [rbp-80h] BYREF
  __int64 v24; // [rsp+28h] [rbp-78h]
  _BYTE v25[112]; // [rsp+30h] [rbp-70h] BYREF

  v2 = *(__int64 **)(a1 + 24);
  sub_DFB180(v2, 1u);
  if ( (unsigned int)sub_DFB120((__int64)v2) || (unsigned int)sub_DFB730(*(_QWORD *)(a1 + 24)) > 1 )
  {
    v5 = *(_QWORD *)(a1 + 16);
    v6 = 0;
    v7 = *(char ***)(v5 + 32);
    for ( i = *(char ***)(v5 + 40); i != v7; v6 |= v10 )
    {
      v9 = *v7++;
      v10 = sub_F6AC10(v9, *(_QWORD *)(a1 + 32), v5, *(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 64), 0, 0);
      v5 = *(_QWORD *)(a1 + 16);
    }
    v11 = *(__int64 **)(v5 + 40);
    v23 = v25;
    v24 = 0x800000000LL;
    if ( v11 == *(__int64 **)(v5 + 32) )
    {
      v15 = v25;
      v14 = 0;
    }
    else
    {
      v12 = *(__int64 **)(v5 + 32);
      while ( 1 )
      {
        v13 = *v12++;
        sub_2ABC990(v13, v5, *(_QWORD *)(a1 + 80), (__int64)&v23);
        if ( v11 == v12 )
          break;
        v5 = *(_QWORD *)(a1 + 16);
      }
      v14 = v24;
      v15 = v23;
    }
    v16 = v6;
    while ( v14 )
    {
      while ( 1 )
      {
        v17 = *(_QWORD *)(a1 + 8);
        v18 = *(_QWORD *)(a1 + 32);
        v19 = *(_QWORD *)&v15[8 * v14 - 8];
        v20 = *(_QWORD *)(a1 + 16);
        LODWORD(v24) = v14 - 1;
        v6 |= sub_11D2180(v19, v18, v20, v17, v3, v4);
        v16 |= sub_2AF1970((unsigned __int8 *)a1, v19);
        LOBYTE(v6) = v16 | v6;
        if ( (_BYTE)v6 )
          break;
        v14 = v24;
        v15 = v23;
        v16 = 0;
        if ( !(_DWORD)v24 )
          goto LABEL_13;
      }
      sub_D37540(*(_QWORD *)(a1 + 72));
      v14 = v24;
      v15 = v23;
    }
LABEL_13:
    v21 = (unsigned __int8)v6;
    BYTE1(v21) = v16;
    if ( v15 != v25 )
      _libc_free((unsigned __int64)v15);
  }
  else
  {
    return 0;
  }
  return v21;
}
