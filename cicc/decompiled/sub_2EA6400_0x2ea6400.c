// Function: sub_2EA6400
// Address: 0x2ea6400
//
_QWORD *__fastcall sub_2EA6400(__int64 a1)
{
  _QWORD *v1; // r14
  _QWORD *v2; // r15
  _QWORD *v3; // rbx
  _QWORD *v4; // rax
  _QWORD *v5; // rdx

  v1 = **(_QWORD ***)(a1 + 32);
  v2 = *(_QWORD **)(v1[4] + 328LL);
  if ( v1 != v2 )
  {
    v3 = (_QWORD *)(*v1 & 0xFFFFFFFFFFFFFFF8LL);
    if ( !*(_BYTE *)(a1 + 84) )
      goto LABEL_9;
LABEL_3:
    v4 = *(_QWORD **)(a1 + 64);
    v5 = &v4[*(unsigned int *)(a1 + 76)];
    if ( v4 != v5 )
    {
      while ( v3 != (_QWORD *)*v4 )
      {
        if ( v5 == ++v4 )
          return v1;
      }
      while ( v3 != v2 )
      {
        v1 = v3;
        v3 = (_QWORD *)(*v3 & 0xFFFFFFFFFFFFFFF8LL);
        if ( *(_BYTE *)(a1 + 84) )
          goto LABEL_3;
LABEL_9:
        if ( !sub_C8CA60(a1 + 56, (__int64)v3) )
          return v1;
      }
      return v3;
    }
  }
  return v1;
}
