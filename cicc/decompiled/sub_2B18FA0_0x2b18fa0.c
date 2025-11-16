// Function: sub_2B18FA0
// Address: 0x2b18fa0
//
__int64 __fastcall sub_2B18FA0(unsigned __int8 *a1, _QWORD *a2, _QWORD *a3, unsigned int a4, __int64 a5)
{
  unsigned __int8 *v9; // rcx
  char *v10; // r15
  __int64 result; // rax
  __int64 v12; // r9
  int v13; // eax
  __int64 v14; // rsi
  int v15; // ecx
  unsigned __int8 *v16; // rdi
  int v17; // r8d
  char v18; // al
  unsigned __int8 *v19; // rbx
  _QWORD *v20; // [rsp+8h] [rbp-48h]

  do
  {
    if ( (a1[7] & 0x40) != 0 )
      v9 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
    else
      v9 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v10 = (char *)*((_QWORD *)v9 + 4);
    v20 = a3;
    result = sub_2B18C70((char *)a1, a4);
    v12 = (unsigned int)result;
    if ( !BYTE4(result) )
      break;
    v13 = *(_DWORD *)(a5 + 2000);
    v14 = *(_QWORD *)(a5 + 1984);
    a3 = v20;
    if ( v13 )
    {
      v15 = v13 - 1;
      result = (v13 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v16 = *(unsigned __int8 **)(v14 + 8 * result);
      if ( a1 == v16 )
        return result;
      v17 = 1;
      while ( v16 != (unsigned __int8 *)-4096LL )
      {
        result = v15 & (unsigned int)(v17 + result);
        v16 = *(unsigned __int8 **)(v14 + 8LL * (unsigned int)result);
        if ( a1 == v16 )
          return result;
        ++v17;
      }
    }
    v18 = *v10;
    if ( (unsigned __int8)*v10 > 0x1Cu && (v18 == 94 || v18 == 91) )
    {
      sub_2B18FA0(v10, a2, v20, (unsigned int)v12, a5);
      a3 = v20;
    }
    else
    {
      *(_QWORD *)(*a2 + 8 * v12) = v10;
      *(_QWORD *)(*v20 + 8 * v12) = a1;
    }
    v19 = (a1[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a1 - 1) : &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    a1 = *(unsigned __int8 **)v19;
    result = *a1;
    if ( (unsigned __int8)result <= 0x1Cu || (_BYTE)result != 91 && (_BYTE)result != 94 )
      break;
    result = *((_QWORD *)a1 + 2);
    if ( !result )
      break;
  }
  while ( !*(_QWORD *)(result + 8) );
  return result;
}
