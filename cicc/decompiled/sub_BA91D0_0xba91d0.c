// Function: sub_BA91D0
// Address: 0xba91d0
//
__int64 __fastcall sub_BA91D0(__int64 a1, const void *a2, size_t a3)
{
  __int64 v3; // r15
  unsigned int v5; // ebx
  __int64 v6; // rdx
  const void *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r13
  unsigned __int8 v11; // al
  unsigned __int8 v13; // al
  __int64 v14; // r13
  int v15; // [rsp+Ch] [rbp-34h]

  v3 = *(_QWORD *)(a1 + 864);
  if ( !v3 )
    return 0;
  v15 = sub_B91A00(v3);
  if ( !v15 )
    return 0;
  v5 = 0;
  while ( 1 )
  {
    v9 = sub_B91A10(v3, v5);
    v10 = v9 - 16;
    v11 = *(_BYTE *)(v9 - 16);
    v6 = (v11 & 2) != 0 ? *(_QWORD *)(v9 - 32) : v10 - 8LL * ((v11 >> 2) & 0xF);
    v7 = (const void *)sub_B91420(*(_QWORD *)(v6 + 8));
    if ( a3 == v8 && (!a3 || !memcmp(a2, v7, a3)) )
      break;
    if ( v15 == ++v5 )
      return 0;
  }
  v13 = *(_BYTE *)(v9 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(v9 - 32);
  else
    v14 = v10 - 8LL * ((v13 >> 2) & 0xF);
  return *(_QWORD *)(v14 + 16);
}
