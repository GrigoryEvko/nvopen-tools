// Function: sub_353FFA0
// Address: 0x353ffa0
//
__int64 __fastcall sub_353FFA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // rdx
  __int64 result; // rax
  __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int8 **v7; // rbx
  __int64 v8; // r13

  v2 = *(_QWORD *)(a1 + 48);
  v3 = (__int64 *)(v2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_15;
  result = v2 & 7;
  if ( (_DWORD)result )
  {
    if ( (_DWORD)result == 3 )
    {
      v3 = (__int64 *)v3[2];
      goto LABEL_4;
    }
LABEL_15:
    BUG();
  }
  *(_QWORD *)(a1 + 48) = v3;
LABEL_4:
  v5 = *v3;
  if ( *v3 )
  {
    if ( (v5 & 4) == 0 )
    {
      v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v6 )
      {
        sub_98B4D0(v6, a2, 0, 6u);
        v7 = *(unsigned __int8 ***)a2;
        result = *(unsigned int *)(a2 + 8);
        v8 = *(_QWORD *)a2 + 8 * result;
        if ( *(_QWORD *)a2 != v8 )
        {
          while ( 1 )
          {
            result = sub_CF7060(*v7);
            if ( !(_BYTE)result )
              break;
            if ( (unsigned __int8 **)v8 == ++v7 )
              return result;
          }
          *(_DWORD *)(a2 + 8) = 0;
        }
      }
    }
  }
  return result;
}
