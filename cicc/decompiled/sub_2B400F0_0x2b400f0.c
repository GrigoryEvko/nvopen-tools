// Function: sub_2B400F0
// Address: 0x2b400f0
//
bool __fastcall sub_2B400F0(__int64 a1, _QWORD ***a2)
{
  __int64 v2; // rax
  _BYTE **v3; // r8
  __int64 v4; // r11
  _QWORD ***v5; // r10
  __int64 *v6; // r9
  __int64 *v7; // rbx
  bool v8; // zf
  _QWORD *v10; // rdi
  _QWORD *v11; // rsi
  __int64 v12; // [rsp-20h] [rbp-20h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_BYTE ***)v2;
  v4 = *(_QWORD *)v2 + 8LL * *(unsigned int *)(v2 + 8);
  if ( *(_QWORD *)v2 == v4 )
    return 1;
  v5 = a2;
  v6 = **(__int64 ***)a1;
  v7 = &v6[*(unsigned int *)(*(_QWORD *)a1 + 8LL)];
  while ( v7 != v6 )
  {
    v8 = **v3 == 12;
    v12 = *v6;
    if ( v8 )
    {
      v10 = **v5;
      v11 = &v10[*((unsigned int *)*v5 + 2)];
      if ( v11 == sub_2B0CA10(v10, (__int64)v11, &v12) )
        break;
    }
    ++v3;
    ++v6;
    if ( (_BYTE **)v4 == v3 )
      return 1;
  }
  return v7 == v6;
}
