// Function: sub_3158930
// Address: 0x3158930
//
void __fastcall sub_3158930(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r12
  __int64 i; // rcx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rbx
  _QWORD *v12; // rax
  __m128i v13; // [rsp+0h] [rbp-40h] BYREF
  char v14; // [rsp+10h] [rbp-30h]

  v7 = a1[2];
  if ( !*(_BYTE *)(v7 - 8) )
    goto LABEL_20;
  while ( 1 )
  {
    while ( 1 )
    {
LABEL_2:
      i = *(_QWORD *)(v7 - 16);
      while ( !i )
      {
        a1[2] -= 24LL;
        v7 = a1[2];
        if ( v7 == a1[1] )
          return;
        if ( *(_BYTE *)(v7 - 8) )
          goto LABEL_2;
LABEL_20:
        for ( i = *(_QWORD *)(*(_QWORD *)(v7 - 24) + 16LL); i; i = *(_QWORD *)(i + 8) )
        {
          if ( (unsigned __int8)(**(_BYTE **)(i + 24) - 30) <= 0xAu )
            break;
        }
        *(_QWORD *)(v7 - 16) = i;
        *(_BYTE *)(v7 - 8) = 1;
      }
      v9 = *(_QWORD *)(i + 8);
      for ( *(_QWORD *)(v7 - 16) = v9; v9; *(_QWORD *)(v7 - 16) = v9 )
      {
        a3 = (unsigned int)**(unsigned __int8 **)(v9 + 24) - 30;
        if ( (unsigned __int8)(**(_BYTE **)(v9 + 24) - 30) <= 0xAu )
          break;
        v9 = *(_QWORD *)(v9 + 8);
      }
      v10 = *a1;
      v11 = *(_QWORD *)(*(_QWORD *)(i + 24) + 40LL);
      if ( !*(_BYTE *)(*a1 + 28) )
        goto LABEL_16;
      v12 = *(_QWORD **)(v10 + 8);
      i = *(unsigned int *)(v10 + 20);
      a3 = (__int64)&v12[i];
      if ( v12 == (_QWORD *)a3 )
        break;
      while ( v11 != *v12 )
      {
        if ( (_QWORD *)a3 == ++v12 )
          goto LABEL_10;
      }
    }
LABEL_10:
    if ( (unsigned int)i < *(_DWORD *)(v10 + 16) )
      break;
LABEL_16:
    sub_C8CC70(v10, v11, a3, i, a5, a6);
    if ( (_BYTE)a3 )
      goto LABEL_12;
  }
  *(_DWORD *)(v10 + 20) = i + 1;
  *(_QWORD *)a3 = v11;
  ++*(_QWORD *)v10;
LABEL_12:
  v13.m128i_i64[0] = v11;
  v14 = 0;
  sub_31588F0(a1 + 1, &v13);
}
