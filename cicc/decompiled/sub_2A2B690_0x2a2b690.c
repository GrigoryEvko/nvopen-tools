// Function: sub_2A2B690
// Address: 0x2a2b690
//
void __fastcall sub_2A2B690(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 v4; // rsi

  if ( (_BYTE)qword_500A928 )
  {
    sub_2A2A680(a1);
    v1 = *(_QWORD *)(*(_QWORD *)(a1 + 280) + 16LL);
    v2 = *(__int64 **)(v1 + 56);
    v3 = &v2[*(unsigned int *)(v1 + 64)];
    while ( v3 != v2 )
    {
      v4 = *v2++;
      sub_2A29CD0(a1, v4, v4);
    }
  }
}
