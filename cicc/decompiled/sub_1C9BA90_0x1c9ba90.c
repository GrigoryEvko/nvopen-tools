// Function: sub_1C9BA90
// Address: 0x1c9ba90
//
void __fastcall sub_1C9BA90(
        _QWORD *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 i; // r15
  _BYTE *v11; // r12
  double v12; // xmm4_8
  double v13; // xmm5_8
  unsigned __int8 v14; // al
  __int64 v15; // rax
  int v16; // edx
  __int64 *v17; // rbx
  __int64 v18; // r13
  char v19; // al
  __int64 v20; // rax
  int v21; // edx
  __int64 *v22; // rbx
  __int64 v23; // r13
  char v24; // al

  for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v11 = sub_1648700(i);
    v14 = v11[16];
    if ( v14 <= 0x17u )
      BUG();
    if ( v14 == 54 )
    {
      v15 = *(_QWORD *)v11;
      if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 13 )
      {
        v16 = *(_DWORD *)(v15 + 12);
        if ( v16 )
        {
          v17 = *(__int64 **)(v15 + 16);
          v18 = (__int64)&v17[(unsigned int)(v16 - 1) + 1];
          while ( 1 )
          {
            v19 = *(_BYTE *)(*v17 + 8);
            if ( v19 == 15 || v19 == 13 && (unsigned __int8)sub_1C97B40(*v17) )
              break;
            if ( (__int64 *)v18 == ++v17 )
              goto LABEL_12;
          }
          sub_1C9B860(a1, (__int64)v11, a3, a4, a5, a6, v12, v13, a9, a10);
        }
      }
    }
    else if ( v14 == 55 )
    {
      v20 = **((_QWORD **)v11 - 6);
      if ( *(_BYTE *)(v20 + 8) == 13 )
      {
        v21 = *(_DWORD *)(v20 + 12);
        if ( v21 )
        {
          v22 = *(__int64 **)(v20 + 16);
          v23 = (__int64)&v22[(unsigned int)(v21 - 1) + 1];
          while ( 1 )
          {
            v24 = *(_BYTE *)(*v22 + 8);
            if ( v24 == 15 || v24 == 13 && (unsigned __int8)sub_1C97B40(*v22) )
              break;
            if ( (__int64 *)v23 == ++v22 )
              goto LABEL_12;
          }
          sub_1C9A550((__int64)a1, (__int64)v11);
        }
      }
    }
LABEL_12:
    ;
  }
}
