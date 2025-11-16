// Function: sub_35F62C0
// Address: 0x35f62c0
//
unsigned __int64 __fastcall sub_35F62C0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // r13
  unsigned __int64 result; // rax
  __int64 v9; // rdx
  void *v10; // rdx

  if ( !a5 )
    sub_C64ED0("Empty modifier in cvt_packfloat Intrinsic.", 1u);
  v7 = *(int *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  result = strlen((const char *)a5);
  if ( result == 4 )
  {
    if ( *(_DWORD *)a5 == 1970038130 && (v7 & 0x200) != 0 )
    {
      v9 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v9;
      if ( result <= 4 )
      {
        return sub_CB6200(a4, ".relu", 5u);
      }
      else
      {
        *(_DWORD *)v9 = 1818587694;
        *(_BYTE *)(v9 + 4) = 117;
        *(_QWORD *)(a4 + 32) += 5LL;
      }
    }
  }
  else if ( result == 3 )
  {
    if ( *(_WORD *)a5 == 28274 && *(_BYTE *)(a5 + 2) == 100 )
    {
      result = BYTE1(v7);
      if ( (v7 & 0x1C00) != 0 )
        result = sub_35ED6D0((v7 >> 10) & 7, a4);
    }
    if ( *(_WORD *)a5 == 24947 && *(_BYTE *)(a5 + 2) == 116 )
    {
      result = (v7 >> 13) & 0xF;
      if ( ((v7 >> 13) & 0xF) != 0 )
      {
        if ( (_BYTE)result != 1 )
          sub_C64ED0("Invalid Saturation Modifier.", 1u);
        v10 = *(void **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v10 <= 9u )
        {
          result = sub_CB6200(a4, ".satfinite", 0xAu);
        }
        else
        {
          qmemcpy(v10, ".satfinite", 10);
          result = 25972;
          *(_QWORD *)(a4 + 32) += 10LL;
        }
      }
    }
    if ( *(_WORD *)a5 == 29540 && *(_BYTE *)(a5 + 2) == 116 )
      result = sub_35ED820(v7 & 0xF, a4);
    if ( *(_WORD *)a5 == 29299 && *(_BYTE *)(a5 + 2) == 99 )
      return sub_35ED820((unsigned __int8)v7 >> 4, a4);
  }
  return result;
}
