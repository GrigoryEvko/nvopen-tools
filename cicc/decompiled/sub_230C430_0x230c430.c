// Function: sub_230C430
// Address: 0x230c430
//
__int64 __fastcall sub_230C430(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rbx
  _QWORD *v3; // r13
  void (__fastcall *v4)(_QWORD *, _QWORD *, __int64); // rax

  *(_QWORD *)a1 = &unk_4A0AF98;
  v1 = *(unsigned int *)(a1 + 32);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 16);
    v3 = &v2[5 * v1];
    do
    {
      if ( *v2 != -8192 && *v2 != -4096 )
      {
        v4 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v2[3];
        if ( v4 )
          v4(v2 + 1, v2 + 1, 3);
      }
      v2 += 5;
    }
    while ( v3 != v2 );
    v1 = *(unsigned int *)(a1 + 32);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 40 * v1, 8);
}
