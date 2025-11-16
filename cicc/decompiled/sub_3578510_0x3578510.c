// Function: sub_3578510
// Address: 0x3578510
//
__int64 __fastcall sub_3578510(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 (*v10)(void); // rax
  __int64 result; // rax
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 (*v14)(); // rax

  v6 = 0;
  v8 = *(_QWORD *)(a1 + 216);
  v9 = *(_QWORD *)(v8 + 32);
  v10 = *(__int64 (**)(void))(**(_QWORD **)(v8 + 16) + 128LL);
  if ( v10 != sub_2DAC790 )
    v6 = v10();
  if ( (a2 & 0x80000000) != 0 )
  {
    result = *(_QWORD *)(v9 + 56) + 16LL * (a2 & 0x7FFFFFFF);
    v12 = *(_QWORD *)(result + 8);
  }
  else
  {
    a3 = *(_QWORD *)(v9 + 304);
    result = a2;
    v12 = *(_QWORD *)(a3 + 8LL * a2);
  }
  if ( !v12 )
    return result;
  if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
  {
LABEL_7:
    v13 = *(_QWORD *)(v12 + 16);
    v14 = *(__int64 (**)())(*(_QWORD *)v6 + 1560LL);
    if ( v14 != sub_2FDC880 )
    {
LABEL_18:
      if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v14)(v6, v13, a2) )
      {
LABEL_9:
        result = *(_QWORD *)(v12 + 16);
        while ( 1 )
        {
          v12 = *(_QWORD *)(v12 + 32);
          if ( !v12 )
            return result;
          if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
          {
            v13 = *(_QWORD *)(v12 + 16);
            if ( result != v13 )
            {
              v14 = *(__int64 (**)())(*(_QWORD *)v6 + 1560LL);
              if ( v14 == sub_2FDC880 )
                break;
              goto LABEL_18;
            }
          }
        }
      }
    }
    sub_3577FF0(a1, v13, a3, a4, a5, a6);
    goto LABEL_9;
  }
  while ( 1 )
  {
    v12 = *(_QWORD *)(v12 + 32);
    if ( !v12 )
      return result;
    if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
      goto LABEL_7;
  }
}
