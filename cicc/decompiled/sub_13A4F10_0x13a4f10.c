// Function: sub_13A4F10
// Address: 0x13a4f10
//
unsigned __int64 __fastcall sub_13A4F10(unsigned __int64 *a1, unsigned int a2, unsigned __int8 a3)
{
  unsigned __int64 result; // rax
  unsigned __int64 v5; // r15
  unsigned int v6; // ebx
  size_t v7; // rdx
  size_t v8; // r8
  char *v9; // r13
  int v10; // ecx
  unsigned int v11; // ebx
  int v12; // ecx
  size_t v13; // [rsp+0h] [rbp-40h]
  size_t n; // [rsp+8h] [rbp-38h]
  size_t na; // [rsp+8h] [rbp-38h]

  *a1 = 1;
  if ( a2 <= 0x39 )
  {
    result = 2 * (((unsigned __int64)a2 << 57) | -(__int64)a3 & ~(-1LL << a2)) + 1;
    *a1 = result;
    return result;
  }
  result = sub_22077B0(24);
  v5 = result;
  if ( result )
  {
    *(_DWORD *)(result + 16) = a2;
    v6 = (a2 + 63) >> 6;
    *(_QWORD *)result = 0;
    *(_QWORD *)(result + 8) = 0;
    result = malloc(8LL * v6);
    v7 = 8LL * v6;
    v8 = v6;
    v9 = (char *)result;
    if ( !result )
    {
      if ( 8LL * v6 || (result = malloc(1u), v8 = v6, v7 = 0, !result) )
      {
        v13 = v7;
        na = v8;
        result = sub_16BD1C0("Allocation failed");
        v8 = na;
        v7 = v13;
      }
      else
      {
        v9 = (char *)result;
      }
    }
    *(_QWORD *)v5 = v9;
    *(_QWORD *)(v5 + 8) = v8;
    if ( v6 )
    {
      n = v8;
      result = (unsigned __int64)memset(v9, -a3, v7);
      if ( !a3 )
        goto LABEL_7;
      v10 = *(_DWORD *)(v5 + 16);
      v11 = (unsigned int)(v10 + 63) >> 6;
      result = v11;
      if ( n > v11 )
      {
        result = (unsigned __int64)memset(&v9[8 * v11], 0, 8 * (n - v11));
        v10 = *(_DWORD *)(v5 + 16);
      }
    }
    else
    {
      if ( !a3 )
        goto LABEL_7;
      v10 = *(_DWORD *)(v5 + 16);
      v11 = (unsigned int)(v10 + 63) >> 6;
    }
    v12 = v10 & 0x3F;
    if ( v12 )
    {
      result = ~(-1LL << v12);
      *(_QWORD *)(*(_QWORD *)v5 + 8LL * (v11 - 1)) &= result;
    }
  }
LABEL_7:
  *a1 = v5;
  return result;
}
