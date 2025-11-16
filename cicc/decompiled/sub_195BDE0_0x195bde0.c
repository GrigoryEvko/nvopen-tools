// Function: sub_195BDE0
// Address: 0x195bde0
//
__int64 __fastcall sub_195BDE0(
        __int64 a1,
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
  __int64 v11; // rbx
  _QWORD *v12; // rax
  _QWORD *v13; // r12
  __int64 result; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // r15
  __int64 v17; // rdi
  __int64 v18; // r15
  __int64 v19; // r12
  unsigned __int64 v20; // r15
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v23; // rbx
  __int64 v24; // rax

  v11 = *(_QWORD *)(a2 + 8);
  if ( v11 )
  {
    while ( 1 )
    {
      v12 = sub_1648700(v11);
      v11 = *(_QWORD *)(v11 + 8);
      v13 = v12;
      if ( (unsigned __int8)(*((_BYTE *)v12 + 16) - 25) <= 9u )
        break;
      if ( !v11 )
        return 0;
    }
    while ( v11 )
    {
      v15 = sub_1648700(v11);
      v11 = *(_QWORD *)(v11 + 8);
      v16 = v15;
      if ( (unsigned __int8)(*((_BYTE *)v15 + 16) - 25) <= 9u )
      {
        if ( v11 )
        {
          while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v11) + 16) - 25) > 9u )
          {
            v11 = *(_QWORD *)(v11 + 8);
            if ( !v11 )
              goto LABEL_11;
          }
          return 0;
        }
LABEL_11:
        v17 = v13[5];
        v18 = v16[5];
        if ( v18 == v17 )
          return 0;
        v19 = sub_157F0B0(v17);
        if ( !v19 )
          return 0;
        if ( v19 != sub_157F0B0(v18) )
          return 0;
        v20 = sub_157EBA0(v19);
        if ( *(_BYTE *)(v20 + 16) != 26 )
          return 0;
        v23 = *(_QWORD *)(a2 + 48);
        if ( a2 + 40 == v23 )
          return 0;
        while ( 1 )
        {
          if ( !v23 )
            BUG();
          if ( *(_BYTE *)(v23 - 8) == 78 )
          {
            v24 = *(_QWORD *)(v23 - 48);
            if ( !*(_BYTE *)(v24 + 16) && *(_DWORD *)(v24 + 36) == 79 )
            {
              result = sub_195ADC0(a1, a2, v23 - 24, v20, a3, a4, a5, a6, v21, v22, a9, a10);
              if ( (_BYTE)result )
                break;
            }
          }
          v23 = *(_QWORD *)(v23 + 8);
          if ( a2 + 40 == v23 )
            return 0;
        }
        return result;
      }
    }
  }
  return 0;
}
