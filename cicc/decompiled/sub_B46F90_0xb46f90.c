// Function: sub_B46F90
// Address: 0xb46f90
//
__int64 __fastcall sub_B46F90(unsigned __int8 *a1, int a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // rcx
  unsigned __int8 *v6; // rcx
  __int64 v7; // rdi
  unsigned __int8 *v8; // rdi

  result = (unsigned int)*a1 - 29;
  switch ( *a1 )
  {
    case 0x1Fu:
      result = (__int64)&a1[-32 * a2 - 32];
      if ( !*(_QWORD *)result )
        goto LABEL_5;
      goto LABEL_3;
    case 0x20u:
      result = *((_QWORD *)a1 - 1) + 32LL * (unsigned int)(2 * a2 + 1);
      if ( !*(_QWORD *)result )
        goto LABEL_5;
      goto LABEL_3;
    case 0x21u:
    case 0x27u:
      result = *((_QWORD *)a1 - 1) + 32LL * (unsigned int)(a2 + 1);
      if ( *(_QWORD *)result )
        goto LABEL_3;
      goto LABEL_5;
    case 0x22u:
      if ( a2 )
      {
        if ( *((_QWORD *)a1 - 8) )
        {
          result = *((_QWORD *)a1 - 7);
          **((_QWORD **)a1 - 6) = result;
          if ( result )
            *(_QWORD *)(result + 16) = *((_QWORD *)a1 - 6);
        }
        *((_QWORD *)a1 - 8) = a3;
        if ( a3 )
        {
          result = *(_QWORD *)(a3 + 16);
          *((_QWORD *)a1 - 7) = result;
          if ( result )
            *(_QWORD *)(result + 16) = a1 - 56;
          *((_QWORD *)a1 - 6) = a3 + 16;
          *(_QWORD *)(a3 + 16) = a1 - 64;
        }
      }
      else
      {
        if ( *((_QWORD *)a1 - 12) )
        {
          result = *((_QWORD *)a1 - 11);
          **((_QWORD **)a1 - 10) = result;
          if ( result )
            *(_QWORD *)(result + 16) = *((_QWORD *)a1 - 10);
        }
        *((_QWORD *)a1 - 12) = a3;
        if ( a3 )
        {
          result = *(_QWORD *)(a3 + 16);
          *((_QWORD *)a1 - 11) = result;
          if ( result )
            *(_QWORD *)(result + 16) = a1 - 88;
          *((_QWORD *)a1 - 10) = a3 + 16;
          *(_QWORD *)(a3 + 16) = a1 - 96;
        }
      }
      return result;
    case 0x25u:
      result = 32 * (1LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF));
      v8 = &a1[result];
      if ( *(_QWORD *)v8 )
      {
        result = *((_QWORD *)v8 + 1);
        **((_QWORD **)v8 + 2) = result;
        if ( result )
          *(_QWORD *)(result + 16) = *((_QWORD *)v8 + 2);
      }
      *(_QWORD *)v8 = a3;
      if ( a3 )
      {
        result = *(_QWORD *)(a3 + 16);
        *((_QWORD *)v8 + 1) = result;
        if ( result )
          *(_QWORD *)(result + 16) = v8 + 8;
        *((_QWORD *)v8 + 2) = a3 + 16;
        *(_QWORD *)(a3 + 16) = v8;
      }
      return result;
    case 0x26u:
      if ( *((_QWORD *)a1 - 4) )
      {
        result = *((_QWORD *)a1 - 3);
        **((_QWORD **)a1 - 2) = result;
        if ( result )
          *(_QWORD *)(result + 16) = *((_QWORD *)a1 - 2);
      }
      *((_QWORD *)a1 - 4) = a3;
      if ( a3 )
      {
        result = *(_QWORD *)(a3 + 16);
        *((_QWORD *)a1 - 3) = result;
        if ( result )
          *(_QWORD *)(result + 16) = a1 - 24;
        *((_QWORD *)a1 - 2) = a3 + 16;
        *(_QWORD *)(a3 + 16) = a1 - 32;
      }
      return result;
    case 0x28u:
      v6 = a1 - 32;
      v7 = *((unsigned int *)a1 + 22);
      if ( a2 )
      {
        result = (__int64)&v6[32 * ((unsigned int)(a2 - 1) - v7)];
        if ( !*(_QWORD *)result )
          goto LABEL_5;
      }
      else
      {
        result = (__int64)&v6[-32 * v7 - 32];
        if ( !*(_QWORD *)result )
          goto LABEL_5;
      }
LABEL_3:
      v4 = *(_QWORD *)(result + 8);
      **(_QWORD **)(result + 16) = v4;
      if ( v4 )
        *(_QWORD *)(v4 + 16) = *(_QWORD *)(result + 16);
LABEL_5:
      *(_QWORD *)result = a3;
      if ( a3 )
      {
        v5 = *(_QWORD *)(a3 + 16);
        *(_QWORD *)(result + 8) = v5;
        if ( v5 )
          *(_QWORD *)(v5 + 16) = result + 8;
        *(_QWORD *)(result + 16) = a3 + 16;
        *(_QWORD *)(a3 + 16) = result;
      }
      return result;
    default:
      BUG();
  }
}
