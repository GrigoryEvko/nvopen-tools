// Function: sub_13D66D0
// Address: 0x13d66d0
//
bool __fastcall sub_13D66D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v5; // al
  __int64 v7; // r13
  unsigned __int8 v8; // al
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r14
  __int64 v15; // r14
  __int64 v16; // r13
  __int64 v17; // r14
  __int64 v18; // r14
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r13
  __int64 v24; // r13
  unsigned int v25; // r15d
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r14
  char v30; // al
  __int64 v31; // r14
  unsigned int v32; // r15d
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r13
  char v37; // al
  __int64 v38; // r13
  int v39; // [rsp+Ch] [rbp-34h]
  int v40; // [rsp+Ch] [rbp-34h]

  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 38 )
  {
    v7 = *(_QWORD *)(a2 - 48);
    v8 = *(_BYTE *)(v7 + 16);
    if ( v8 == 14 )
    {
      if ( *(_QWORD *)(v7 + 32) == sub_16982C0(a1, a2, a3, a4) )
        v9 = *(_QWORD *)(v7 + 40) + 8LL;
      else
        v9 = v7 + 32;
      if ( (*(_BYTE *)(v9 + 18) & 7) != 3 )
        return 0;
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 || v8 > 0x10u )
        return 0;
      v10 = *(_QWORD *)(a2 - 48);
      v11 = sub_15A1020(v10);
      v14 = v11;
      if ( v11 && *(_BYTE *)(v11 + 16) == 14 )
      {
        if ( *(_QWORD *)(v11 + 32) == sub_16982C0(v10, a2, v12, v13) )
          v15 = *(_QWORD *)(v14 + 40) + 8LL;
        else
          v15 = v14 + 32;
        if ( (*(_BYTE *)(v15 + 18) & 7) != 3 )
          return 0;
      }
      else
      {
        v39 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
        if ( v39 )
        {
          v25 = 0;
          while ( 1 )
          {
            v26 = sub_15A0A60(v7, v25);
            v29 = v26;
            if ( !v26 )
              break;
            v30 = *(_BYTE *)(v26 + 16);
            if ( v30 != 9 )
            {
              if ( v30 != 14 )
                break;
              v31 = *(_QWORD *)(v29 + 32) == sub_16982C0(v7, v25, v27, v28) ? *(_QWORD *)(v29 + 40) + 8LL : v29 + 32;
              if ( (*(_BYTE *)(v31 + 18) & 7) != 3 )
                break;
            }
            if ( v39 == ++v25 )
              return *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 - 24);
          }
          return 0;
        }
      }
    }
    return *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 - 24);
  }
  else
  {
    if ( v5 != 5 || *(_WORD *)(a2 + 18) != 14 )
      return 0;
    v16 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v17 = *(_QWORD *)(a2 - 24 * v16);
    if ( *(_BYTE *)(v17 + 16) == 14 )
    {
      if ( *(_QWORD *)(v17 + 32) == sub_16982C0(a1, a2, 4 * v16, a4) )
        v18 = *(_QWORD *)(v17 + 40) + 8LL;
      else
        v18 = v17 + 32;
      if ( (*(_BYTE *)(v18 + 18) & 7) != 3 )
        return 0;
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v17 + 8LL) != 16 )
        return 0;
      v19 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v20 = sub_15A1020(v19);
      v23 = v20;
      if ( v20 && *(_BYTE *)(v20 + 16) == 14 )
      {
        if ( *(_QWORD *)(v20 + 32) == sub_16982C0(v19, a2, v21, v22) )
          v24 = *(_QWORD *)(v23 + 40) + 8LL;
        else
          v24 = v23 + 32;
        if ( (*(_BYTE *)(v24 + 18) & 7) != 3 )
          return 0;
      }
      else
      {
        v40 = *(_QWORD *)(*(_QWORD *)v17 + 32LL);
        if ( v40 )
        {
          v32 = 0;
          do
          {
            v33 = sub_15A0A60(v17, v32);
            v36 = v33;
            if ( !v33 )
              return 0;
            v37 = *(_BYTE *)(v33 + 16);
            if ( v37 != 9 )
            {
              if ( v37 != 14 )
                return 0;
              v38 = *(_QWORD *)(v36 + 32) == sub_16982C0(v17, v32, v34, v35) ? *(_QWORD *)(v36 + 40) + 8LL : v36 + 32;
              if ( (*(_BYTE *)(v38 + 18) & 7) != 3 )
                return 0;
            }
          }
          while ( v40 != ++v32 );
        }
      }
      v16 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
    return *(_QWORD *)(a2 + 24 * (1 - v16)) == *(_QWORD *)(a1 + 8);
  }
}
