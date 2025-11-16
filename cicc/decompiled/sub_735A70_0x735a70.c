// Function: sub_735A70
// Address: 0x735a70
//
__int64 __fastcall sub_735A70(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // rbx
  __int64 v3; // rcx
  _QWORD *v4; // rax
  _QWORD *v5; // rdx

  if ( *(_BYTE *)(a1 + 177) == 2 )
  {
    v2 = *(_QWORD **)(a1 + 184);
    sub_734850((__int64)v2);
    v3 = unk_4D03FF0;
    v4 = *(_QWORD **)(unk_4F07288 + 192LL);
    if ( v4 == v2 )
    {
      *(_QWORD *)(unk_4F07288 + 192LL) = *v2;
      v5 = 0;
    }
    else
    {
      do
      {
        v5 = v4;
        v4 = (_QWORD *)*v4;
      }
      while ( v4 != v2 );
      *v5 = *v2;
    }
    if ( !*v2 )
      *(_QWORD *)(v3 + 88) = v5;
    *v2 = 0;
  }
  *(_DWORD *)(a1 + 174) &= 0xFFFF1Fu;
  result = sub_86A990(a1);
  if ( !*(_BYTE *)(a1 + 136) )
  {
    *(_BYTE *)(a1 + 136) = 1;
    *(_QWORD *)(a1 + 240) = 0;
  }
  if ( *(char *)(a1 + 170) >= 0 )
    return sub_8C3FF0(a1);
  return result;
}
