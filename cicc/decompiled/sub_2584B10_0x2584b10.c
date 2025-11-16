// Function: sub_2584B10
// Address: 0x2584b10
//
__int64 __fastcall sub_2584B10(__int64 *a1, unsigned __int64 a2, _BYTE *a3)
{
  unsigned __int64 v3; // rax
  unsigned __int8 *v5; // r12
  int v6; // edi
  unsigned int v7; // ecx
  unsigned int v8; // r13d
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdi
  int v20; // r13d
  _QWORD v21[8]; // [rsp+10h] [rbp-40h] BYREF

  LOBYTE(v3) = 0;
  v5 = *(unsigned __int8 **)(a2 + 24);
  v6 = *v5;
  v7 = v6 - 63;
  if ( (unsigned __int8)(v6 - 63) <= 0x17u )
    v3 = ((unsigned __int64)&loc_A1FFF1 >> v7) & 1;
  if ( (unsigned __int8)v3 | ((unsigned __int8)v6 <= 0x1Cu) )
  {
    *a3 = 1;
    LOBYTE(v7) = v3 | ((unsigned __int8)v6 <= 0x1Cu);
    return v7;
  }
  else
  {
    LOBYTE(v7) = (_BYTE)v6 == 61 || (unsigned __int8)(v6 - 82) <= 1u;
    v8 = v7;
    if ( !(_BYTE)v7 )
    {
      if ( (_BYTE)v6 == 62 )
      {
        LOBYTE(v8) = *((_QWORD *)v5 - 8) != *(_QWORD *)a2;
      }
      else
      {
        v10 = (unsigned int)(v6 - 34);
        if ( (unsigned __int8)v10 <= 0x33u )
        {
          v11 = 0x8000000000041LL;
          if ( _bittest64(&v11, v10) )
          {
            v12 = (_BYTE *)*((_QWORD *)v5 - 4);
            if ( !v12 || *v12 || (v12[32] & 0xFu) - 7 > 1 )
            {
              return 1;
            }
            else if ( sub_254C190(*(unsigned __int8 **)(a2 + 24), a2) )
            {
              v13 = *a1;
              v14 = sub_254C9B0((__int64)v5, (__int64)(a2 - (_QWORD)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)]) >> 5);
              v16 = sub_252B790(v13, v14, v15, a1[1], 1, 0, 1);
              if ( v16 )
              {
                if ( *(_BYTE *)(v16 + 97) )
                {
                  v17 = (__int64 *)a1[2];
                  v18 = a1[1];
                  v19 = *a1;
                  v21[0] = *v17;
                  v21[3] = sub_2535070;
                  v21[2] = sub_2535AE0;
                  v20 = sub_2529400(v19, (__int64)v5, *v17, v18, 0, (__int64)v21);
                  sub_A17130((__int64)v21);
                  return v20 ^ 1u;
                }
              }
            }
          }
        }
      }
    }
    return v8;
  }
}
