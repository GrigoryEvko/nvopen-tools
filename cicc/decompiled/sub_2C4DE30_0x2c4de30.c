// Function: sub_2C4DE30
// Address: 0x2c4de30
//
bool __fastcall sub_2C4DE30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 **a7,
        unsigned __int8 **a8,
        __int64 a9)
{
  __int64 i; // rbx
  unsigned __int8 *v11; // r12
  __int64 *v13; // rax
  unsigned __int8 **v14; // rax
  unsigned __int8 **v15; // rdx
  char v16; // al
  __int64 v17; // [rsp+8h] [rbp-38h]
  __int64 v18; // [rsp+8h] [rbp-38h]

  for ( i = a1; a2 != i; i = *(_QWORD *)(i + 8) )
  {
    v11 = *(unsigned __int8 **)(i + 24);
    if ( v11 == *a7 || v11 == *a8 )
      continue;
    if ( *v11 != 92 )
      return a2 != i;
    if ( *(_BYTE *)(a9 + 28) )
    {
      v14 = *(unsigned __int8 ***)(a9 + 8);
      v15 = &v14[*(unsigned int *)(a9 + 20)];
      if ( v14 != v15 )
      {
        while ( v11 != *v14 )
        {
          if ( v15 == ++v14 )
            goto LABEL_15;
        }
        continue;
      }
    }
    else
    {
      v17 = a9;
      v13 = sub_C8CA60(a9, *(_QWORD *)(i + 24));
      a9 = v17;
      if ( v13 )
        continue;
    }
LABEL_15:
    v18 = a9;
    v16 = sub_F50EE0(v11, 0);
    a9 = v18;
    if ( !v16 )
      return a2 != i;
  }
  return a2 != i;
}
