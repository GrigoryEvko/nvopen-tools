// Function: sub_15A0C20
// Address: 0x15a0c20
//
_BOOL8 __fastcall sub_15A0C20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rdi
  __int64 v10; // r12
  char v11; // al
  char v12; // al
  int v14; // r14d
  unsigned int v15; // r13d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r12
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r15
  char v26; // al

  if ( *(_BYTE *)(a1 + 16) != 14 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
    {
      v14 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( v14 )
      {
        v15 = 0;
        while ( 1 )
        {
          v16 = sub_15A0A60(a1, v15);
          v19 = v16;
          if ( !v16 || *(_BYTE *)(v16 + 16) != 14 )
            break;
          v20 = v16 + 32;
          v24 = sub_16982C0(a1, v15, v17, v18);
          if ( *(_QWORD *)(v19 + 32) == v24
             ? sub_16A0F40(v20, v15, v21, v22, v23)
             : (unsigned __int8)sub_16984B0(v20, v15, v21, v22, v23) )
          {
            break;
          }
          if ( v24 == *(_QWORD *)(v19 + 32) )
          {
            v26 = *(_BYTE *)(*(_QWORD *)(v19 + 40) + 26LL) & 7;
            if ( v26 == 1 )
              return 0;
          }
          else
          {
            v26 = *(_BYTE *)(v19 + 50) & 7;
            if ( v26 == 1 )
              return 0;
          }
          if ( v26 == 3 || !v26 )
            break;
          if ( v14 == ++v15 )
            return 1;
        }
        return 0;
      }
      return 1;
    }
    return 0;
  }
  v5 = sub_16982C0(a1, a2, a3, a4);
  v9 = a1 + 32;
  v10 = v5;
  if ( *(_QWORD *)(a1 + 32) == v5 )
    v11 = sub_16A0F40(v9, a2, v6, v7, v8);
  else
    v11 = sub_16984B0(v9, a2, v6, v7, v8);
  if ( v11 )
    return 0;
  if ( v10 == *(_QWORD *)(a1 + 32) )
  {
    v12 = *(_BYTE *)(*(_QWORD *)(a1 + 40) + 26LL) & 7;
    if ( v12 != 1 )
      return v12 != 3 && v12;
    return 0;
  }
  v12 = *(_BYTE *)(a1 + 50) & 7;
  if ( v12 == 1 )
    return 0;
  return v12 != 3 && v12;
}
