// Function: sub_38D6F80
// Address: 0x38d6f80
//
__int64 __fastcall sub_38D6F80(_QWORD *a1, unsigned int a2, int a3, __int64 a4)
{
  __int16 *v5; // rdx
  __int16 v6; // ax
  __int16 *v7; // rdx
  unsigned __int16 v8; // r14
  __int16 *v9; // rbx
  unsigned int v10; // r15d
  __int64 v11; // rax
  int v12; // edx
  __int16 v13; // ax

  v5 = (__int16 *)(a1[6] + 2LL * *(unsigned int *)(*a1 + 24LL * a2 + 8));
  v6 = *v5;
  v7 = v5 + 1;
  v8 = v6 + a2;
  if ( !v6 )
    v7 = 0;
LABEL_3:
  v9 = v7;
  if ( v7 )
  {
    while ( 1 )
    {
      v10 = v8;
      v11 = v8 >> 3;
      if ( (unsigned int)v11 < *(unsigned __int16 *)(a4 + 22) )
      {
        v12 = *(unsigned __int8 *)(*(_QWORD *)(a4 + 8) + v11);
        if ( _bittest(&v12, v8 & 7) )
        {
          if ( (unsigned int)sub_38D6F10(a1, v8, a3) == a2 )
            break;
        }
      }
      v13 = *v9;
      v7 = 0;
      ++v9;
      if ( !v13 )
        goto LABEL_3;
      v8 += v13;
      if ( !v9 )
        return 0;
    }
  }
  else
  {
    return 0;
  }
  return v10;
}
