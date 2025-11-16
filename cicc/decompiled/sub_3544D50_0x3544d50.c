// Function: sub_3544D50
// Address: 0x3544d50
//
__int64 __fastcall sub_3544D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned int v5; // r14d
  unsigned __int64 v8; // rax
  __int64 v9; // r15
  int v10; // eax
  _BYTE *v11; // rax
  _BYTE *v12; // r12
  _BYTE *v13; // rbx
  _BYTE *v14; // r15
  _BOOL4 v15; // r13d
  _BYTE *v16; // rbx
  int v17; // [rsp+Ch] [rbp-34h]

  v5 = 0;
  if ( !*(_BYTE *)a4 )
  {
    LOBYTE(a5) = *(_WORD *)(a3 + 68) == 0 || *(_WORD *)(a3 + 68) == 68;
    v5 = a5;
    if ( (_BYTE)a5 )
    {
      return 0;
    }
    else
    {
      v8 = sub_2EBEE10(*(_QWORD *)(a1 + 104), *(_DWORD *)(a4 + 8));
      v9 = v8;
      if ( v8 )
      {
        v10 = *(unsigned __int16 *)(v8 + 68);
        if ( (v10 == 68 || !v10) && *(_QWORD *)(v9 + 24) == *(_QWORD *)(a3 + 24) )
        {
          if ( (unsigned __int8)sub_3544A40(a1, a2, v9) )
          {
            v17 = sub_353D010(v9, *(_QWORD *)(v9 + 24));
            v11 = *(_BYTE **)(a3 + 32);
            v12 = &v11[40 * (*(_DWORD *)(a3 + 40) & 0xFFFFFF)];
            if ( v11 != v12 )
            {
              v13 = *(_BYTE **)(a3 + 32);
              while ( 1 )
              {
                v14 = v13;
                v15 = sub_2DADC00(v13);
                if ( v15 )
                  break;
                v13 += 40;
                if ( v12 == v13 )
                  return v5;
              }
              while ( v12 != v14 )
              {
                if ( *((_DWORD *)v14 + 2) == v17 )
                  return v15;
                v16 = v14 + 40;
                if ( v14 + 40 == v12 )
                  return v5;
                while ( 1 )
                {
                  v14 = v16;
                  if ( sub_2DADC00(v16) )
                    break;
                  v16 += 40;
                  if ( v12 == v16 )
                    return v5;
                }
              }
            }
          }
        }
      }
    }
  }
  return v5;
}
