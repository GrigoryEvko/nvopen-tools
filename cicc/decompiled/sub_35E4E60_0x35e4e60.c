// Function: sub_35E4E60
// Address: 0x35e4e60
//
unsigned __int64 __fastcall sub_35E4E60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // r11
  signed __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // r10
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r9
  __int64 v17; // r10
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rcx

  v3 = a3;
  if ( a1 == a2 )
    return v3;
  v4 = a1;
  if ( a2 != a3 )
  {
    v3 = a1 + a3 - a2;
    v5 = 0xAAAAAAAAAAAAAAABLL * ((a3 - a1) >> 3);
    v6 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3);
    if ( v6 == v5 - v6 )
    {
      v19 = a2;
      v20 = v4;
      do
      {
        v21 = *(_QWORD *)(v19 + 16);
        v22 = *(_QWORD *)(v20 + 16);
        v20 += 24;
        v19 += 24;
        *(_QWORD *)(v20 - 8) = v21;
        LODWORD(v21) = *(_DWORD *)(v19 - 16);
        *(_QWORD *)(v19 - 8) = v22;
        LODWORD(v22) = *(_DWORD *)(v20 - 16);
        *(_DWORD *)(v20 - 16) = v21;
        LODWORD(v21) = *(_DWORD *)(v19 - 20);
        *(_DWORD *)(v19 - 16) = v22;
        LODWORD(v22) = *(_DWORD *)(v20 - 20);
        *(_DWORD *)(v20 - 20) = v21;
        LODWORD(v21) = *(_DWORD *)(v19 - 24);
        *(_DWORD *)(v19 - 20) = v22;
        LODWORD(v22) = *(_DWORD *)(v20 - 24);
        *(_DWORD *)(v20 - 24) = v21;
        *(_DWORD *)(v19 - 24) = v22;
      }
      while ( a2 != v20 );
      return v4 + 8 * ((unsigned __int64)(a2 - 24 - v4) >> 3) + 24;
    }
    else
    {
      v7 = v5 - v6;
      if ( v6 >= v5 - v6 )
        goto LABEL_12;
      while ( 1 )
      {
        v8 = v4 + 24 * v6;
        if ( v7 > 0 )
        {
          v9 = v4;
          v10 = 0;
          do
          {
            v11 = *(_QWORD *)(v9 + 16);
            v12 = *(_QWORD *)(v8 + 16);
            ++v10;
            v9 += 24;
            v8 += 24;
            *(_QWORD *)(v9 - 8) = v12;
            LODWORD(v12) = *(_DWORD *)(v8 - 16);
            *(_QWORD *)(v8 - 8) = v11;
            LODWORD(v11) = *(_DWORD *)(v9 - 16);
            *(_DWORD *)(v9 - 16) = v12;
            LODWORD(v12) = *(_DWORD *)(v8 - 20);
            *(_DWORD *)(v8 - 16) = v11;
            LODWORD(v11) = *(_DWORD *)(v9 - 20);
            *(_DWORD *)(v9 - 20) = v12;
            LODWORD(v12) = *(_DWORD *)(v8 - 24);
            *(_DWORD *)(v8 - 20) = v11;
            LODWORD(v11) = *(_DWORD *)(v9 - 24);
            *(_DWORD *)(v9 - 24) = v12;
            *(_DWORD *)(v8 - 24) = v11;
          }
          while ( v7 != v10 );
          v4 += 24 * v7;
        }
        if ( !(v5 % v6) )
          break;
        v7 = v6;
        v6 -= v5 % v6;
        while ( 1 )
        {
          v5 = v7;
          v7 -= v6;
          if ( v6 < v7 )
            break;
LABEL_12:
          v13 = v4 + 24 * v5;
          v4 = v13 - 24 * v7;
          if ( v6 > 0 )
          {
            v14 = v13 - 24 * v7;
            v15 = 0;
            do
            {
              v16 = *(_QWORD *)(v14 - 8);
              v17 = *(_QWORD *)(v13 - 8);
              ++v15;
              v14 -= 24;
              v13 -= 24;
              *(_QWORD *)(v14 + 16) = v17;
              LODWORD(v17) = *(_DWORD *)(v13 + 8);
              *(_QWORD *)(v13 + 16) = v16;
              LODWORD(v16) = *(_DWORD *)(v14 + 8);
              *(_DWORD *)(v14 + 8) = v17;
              LODWORD(v17) = *(_DWORD *)(v13 + 4);
              *(_DWORD *)(v13 + 8) = v16;
              LODWORD(v16) = *(_DWORD *)(v14 + 4);
              *(_DWORD *)(v14 + 4) = v17;
              LODWORD(v17) = *(_DWORD *)v13;
              *(_DWORD *)(v13 + 4) = v16;
              LODWORD(v16) = *(_DWORD *)v14;
              *(_DWORD *)v14 = v17;
              *(_DWORD *)v13 = v16;
            }
            while ( v6 != v15 );
            v4 -= 24 * v6;
          }
          v6 = v5 % v7;
          if ( !(v5 % v7) )
            return v3;
        }
      }
    }
    return v3;
  }
  return a1;
}
