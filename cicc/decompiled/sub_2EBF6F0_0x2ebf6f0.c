// Function: sub_2EBF6F0
// Address: 0x2ebf6f0
//
__int64 __fastcall sub_2EBF6F0(_QWORD *a1, unsigned int a2, char a3)
{
  unsigned int v3; // r8d
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rsi
  _QWORD *v10; // rax
  char *v11; // rax
  __int64 v12; // rdx
  unsigned __int16 *v13; // rcx
  char *v14; // r12
  __int64 v15; // r14
  __int64 v16; // r15
  int v17; // eax
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  _BYTE *v22; // r15
  unsigned __int16 *v24; // [rsp+8h] [rbp-38h]
  unsigned __int16 *v25; // [rsp+8h] [rbp-38h]

  v3 = 1;
  v6 = a2;
  v7 = a2 >> 6;
  v8 = 1LL << a2;
  v9 = a1[39];
  if ( (*(_QWORD *)(v9 + 8 * v7) & v8) == 0 )
  {
    v10 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(**(_QWORD **)(*a1 + 16LL)
                                                                                          + 200LL))(
                      *(_QWORD *)(*a1 + 16LL),
                      v9,
                      v7,
                      v6,
                      1);
    v11 = sub_E922F0(v10, a2);
    v13 = (unsigned __int16 *)&v11[2 * v12];
    v14 = v11;
    if ( v11 == (char *)v13 )
    {
      return 0;
    }
    else
    {
      while ( 1 )
      {
        v15 = *(_QWORD *)(a1[38] + 8LL * *(unsigned __int16 *)v14);
        if ( v15 )
        {
          if ( (*(_BYTE *)(v15 + 3) & 0x10) != 0 )
            break;
          v15 = *(_QWORD *)(v15 + 32);
          if ( v15 )
          {
            if ( (*(_BYTE *)(v15 + 3) & 0x10) != 0 )
              break;
          }
        }
LABEL_23:
        v14 += 2;
        if ( v13 == (unsigned __int16 *)v14 )
          return 0;
      }
      if ( !a3 )
      {
        while ( 1 )
        {
          v16 = *(_QWORD *)(v15 + 16);
          v17 = *(_DWORD *)(v16 + 44);
          if ( (v17 & 4) == 0 && (v17 & 8) != 0 )
          {
            v25 = v13;
            v18 = sub_2E88A90(*(_QWORD *)(v15 + 16), 128, 1);
            v13 = v25;
          }
          else
          {
            v18 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v16 + 16) + 24LL) >> 7;
          }
          v24 = v13;
          if ( !v18 )
            break;
          v19 = *(_QWORD *)(v16 + 24);
          if ( *(_DWORD *)(v19 + 120) )
            break;
          if ( (unsigned __int8)sub_B2D610(**(_QWORD **)(v19 + 32), 95) )
            break;
          v20 = *(_QWORD *)(v16 + 32);
          v21 = v20 + 40LL * (*(_DWORD *)(v16 + 40) & 0xFFFFFF);
          if ( v20 == v21 )
            break;
          while ( 1 )
          {
            if ( *(_BYTE *)v20 == 10 )
            {
              v22 = *(_BYTE **)(v20 + 24);
              if ( !*v22 )
                break;
            }
            v20 += 40;
            if ( v21 == v20 )
              return 1;
          }
          if ( !(unsigned __int8)sub_B2D610(*(_QWORD *)(v20 + 24), 36) || !(unsigned __int8)sub_B2D610((__int64)v22, 41) )
            break;
          v15 = *(_QWORD *)(v15 + 32);
          v13 = v24;
          if ( !v15 || (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
            goto LABEL_23;
        }
      }
      return 1;
    }
  }
  return v3;
}
