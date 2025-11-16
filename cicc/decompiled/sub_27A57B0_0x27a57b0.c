// Function: sub_27A57B0
// Address: 0x27a57b0
//
void __fastcall sub_27A57B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 i; // rcx
  __int64 v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // rax
  __m128i v11; // [rsp+0h] [rbp-40h] BYREF
  char v12; // [rsp+10h] [rbp-30h]

  v6 = *(_QWORD *)(a1 + 104);
  if ( !*(_BYTE *)(v6 - 8) )
    goto LABEL_20;
  while ( 1 )
  {
    while ( 1 )
    {
LABEL_2:
      i = *(_QWORD *)(v6 - 16);
      while ( !i )
      {
        *(_QWORD *)(a1 + 104) -= 24LL;
        v6 = *(_QWORD *)(a1 + 104);
        if ( v6 == *(_QWORD *)(a1 + 96) )
          return;
        if ( *(_BYTE *)(v6 - 8) )
          goto LABEL_2;
LABEL_20:
        for ( i = *(_QWORD *)(*(_QWORD *)(v6 - 24) + 16LL); i; i = *(_QWORD *)(i + 8) )
        {
          if ( (unsigned __int8)(**(_BYTE **)(i + 24) - 30) <= 0xAu )
            break;
        }
        *(_QWORD *)(v6 - 16) = i;
        *(_BYTE *)(v6 - 8) = 1;
      }
      v8 = *(_QWORD *)(i + 8);
      *(_QWORD *)(v6 - 16) = v8;
      if ( v8 )
      {
        while ( 1 )
        {
          a3 = (unsigned int)**(unsigned __int8 **)(v8 + 24) - 30;
          if ( (unsigned __int8)(**(_BYTE **)(v8 + 24) - 30) <= 0xAu )
            break;
          v8 = *(_QWORD *)(v8 + 8);
          *(_QWORD *)(v6 - 16) = v8;
          if ( !v8 )
          {
            v9 = *(_QWORD *)(*(_QWORD *)(i + 24) + 40LL);
            if ( *(_BYTE *)(a1 + 28) )
              goto LABEL_7;
            goto LABEL_16;
          }
        }
      }
      v9 = *(_QWORD *)(*(_QWORD *)(i + 24) + 40LL);
      if ( !*(_BYTE *)(a1 + 28) )
        goto LABEL_16;
LABEL_7:
      v10 = *(_QWORD **)(a1 + 8);
      i = *(unsigned int *)(a1 + 20);
      a3 = (__int64)&v10[i];
      if ( v10 == (_QWORD *)a3 )
        break;
      while ( v9 != *v10 )
      {
        if ( (_QWORD *)a3 == ++v10 )
          goto LABEL_10;
      }
    }
LABEL_10:
    if ( (unsigned int)i < *(_DWORD *)(a1 + 16) )
      break;
LABEL_16:
    sub_C8CC70(a1, v9, a3, i, a5, a6);
    if ( (_BYTE)a3 )
      goto LABEL_12;
  }
  *(_DWORD *)(a1 + 20) = i + 1;
  *(_QWORD *)a3 = v9;
  ++*(_QWORD *)a1;
LABEL_12:
  v11.m128i_i64[0] = v9;
  v12 = 0;
  sub_27A5770((unsigned __int64 *)(a1 + 96), &v11);
}
