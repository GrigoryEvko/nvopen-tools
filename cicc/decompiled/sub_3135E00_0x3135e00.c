// Function: sub_3135E00
// Address: 0x3135e00
//
__int64 __fastcall sub_3135E00(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  int v5; // eax
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // r15
  char v15; // [rsp+7h] [rbp-49h] BYREF
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  __int64 v17; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v18; // [rsp+18h] [rbp-38h]

  v2 = sub_B2D810(a2, "omp_target_thread_limit", 0x17u, 0);
  v3 = (unsigned int)v2;
  if ( (unsigned int)(*(_DWORD *)(a1 + 32) - 26) <= 1 )
  {
    v16 = sub_B2D7E0(a2, "amdgpu-flat-work-group-size", 0x1Bu);
    if ( v16 && sub_A71840((__int64)&v16) )
    {
      v6 = sub_A72240(&v16);
      v18 = v7;
      v17 = v6;
      v15 = 44;
      v8 = sub_C931B0(&v17, &v15, 1u, 0);
      if ( v8 == -1 )
      {
        v11 = v17;
        v9 = v18;
        v12 = 0;
        v13 = 0;
      }
      else
      {
        v9 = v18;
        v10 = v8 + 1;
        v11 = v17;
        if ( v8 + 1 > v18 )
        {
          v10 = v18;
          v12 = 0;
        }
        else
        {
          v12 = v18 - v10;
        }
        v13 = v17 + v10;
        if ( v8 <= v18 )
          v9 = v8;
      }
      if ( !sub_C93CC0(v13, v12, 0xAu, &v17) )
      {
        v14 = (unsigned int)v17;
        if ( v17 == (int)v17 )
        {
          if ( (_DWORD)v2 )
          {
            v14 = (unsigned int)v2;
            if ( (int)v17 <= (int)v2 )
              v14 = (unsigned int)v17;
          }
          if ( sub_C93CC0(v11, v9, 0xAu, &v17) || v17 != (int)v17 )
            return v14 << 32;
          else
            return (v14 << 32) | (unsigned int)v17;
        }
      }
    }
    return v2 << 32;
  }
  if ( !(unsigned __int8)sub_B2D620(a2, "nvvm.maxntid", 0xCu) )
    return v2 << 32;
  v5 = sub_B2D810(a2, "nvvm.maxntid", 0xCu, 0);
  if ( !(_DWORD)v2 || (int)v2 > v5 )
    v3 = (unsigned int)v5;
  return v3 << 32;
}
