// Function: sub_AF4B80
// Address: 0xaf4b80
//
__int64 __fastcall sub_AF4B80(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 *v3; // r15
  unsigned __int64 *v4; // r13
  unsigned __int64 v5; // r14
  unsigned __int64 *v6; // r15
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  size_t v10; // r12
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rdx
  unsigned __int8 v14; // [rsp+7h] [rbp-69h]
  unsigned __int64 *v16; // [rsp+18h] [rbp-58h] BYREF
  void *src[2]; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int8 v18; // [rsp+30h] [rbp-40h]

  *a2 = 0;
  *(_DWORD *)(a3 + 8) = 0;
  sub_AF4640((__int64)src, a1);
  v14 = v18;
  if ( v18 )
  {
    v3 = (unsigned __int64 *)src[0];
    v4 = (unsigned __int64 *)((char *)src[0] + 8 * (__int64)src[1]);
    v16 = (unsigned __int64 *)src[0];
    if ( v4 == src[0] )
    {
LABEL_23:
      v10 = (char *)v4 - (char *)v3;
      v11 = *(unsigned int *)(a3 + 8);
      v12 = v4 - v3;
      v13 = v11 + v12;
      if ( *(unsigned int *)(a3 + 12) >= (unsigned __int64)(v11 + v12) )
      {
LABEL_22:
        *(_DWORD *)(a3 + 8) = v12 + v11;
        return v14;
      }
    }
    else
    {
      while ( 1 )
      {
        v7 = *v3;
        if ( *v3 == 6 || v7 == 148 || v7 == 166 || v7 == 4096 || v7 - 4102 <= 1 )
          break;
        if ( v7 == 35 )
        {
          v8 = v3[1];
          v6 = v16;
          *a2 += v8;
        }
        else
        {
          if ( v7 != 16 )
            return 0;
          v5 = v3[1];
          v6 = &v3[(unsigned int)sub_AF4160(&v16)];
          v16 = v6;
          if ( *v6 == 34 )
          {
            *a2 += v5;
          }
          else
          {
            if ( *v6 != 28 )
              return 0;
            *a2 -= v5;
          }
        }
        v3 = &v6[(unsigned int)sub_AF4160(&v16)];
        v16 = v3;
        if ( v4 == v3 )
          goto LABEL_23;
      }
      v10 = (char *)v4 - (char *)v3;
      v11 = *(unsigned int *)(a3 + 8);
      v12 = v4 - v3;
      v13 = v12 + v11;
      if ( v12 + v11 <= (unsigned __int64)*(unsigned int *)(a3 + 12) )
        goto LABEL_20;
    }
    sub_C8D5F0(a3, a3 + 16, v13, 8);
    v11 = *(unsigned int *)(a3 + 8);
LABEL_20:
    if ( v4 != v3 )
    {
      memcpy((void *)(*(_QWORD *)a3 + 8 * v11), v3, v10);
      LODWORD(v11) = *(_DWORD *)(a3 + 8);
    }
    goto LABEL_22;
  }
  return 0;
}
