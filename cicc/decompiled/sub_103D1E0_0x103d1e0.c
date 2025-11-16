// Function: sub_103D1E0
// Address: 0x103d1e0
//
__int64 __fastcall sub_103D1E0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v4; // r13
  __int64 v5; // rax
  void *v6; // rdx
  unsigned __int64 v7; // rsi
  _BYTE *v8; // rax
  __int64 result; // rax
  int v10; // edx
  void *v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rsi

  v2 = (__int64 *)(a1 - 64);
  if ( *(_BYTE *)a1 == 26 )
    v2 = (__int64 *)(a1 - 32);
  v4 = *v2;
  v5 = sub_CB59D0(a2, *(unsigned int *)(a1 + 80));
  v6 = *(void **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xCu )
  {
    sub_CB6200(v5, " = MemoryDef(", 0xDu);
  }
  else
  {
    qmemcpy(v6, " = MemoryDef(", 13);
    *(_QWORD *)(v5 + 32) += 13LL;
  }
  if ( v4 && (*(_BYTE *)v4 != 27 ? (v7 = *(unsigned int *)(v4 + 72)) : (v7 = *(unsigned int *)(v4 + 80)), (_DWORD)v7) )
  {
    sub_CB59D0(a2, v7);
    v8 = *(_BYTE **)(a2 + 32);
  }
  else
  {
    v11 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v11 > 0xAu )
    {
      qmemcpy(v11, "liveOnEntry", 11);
      v8 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 11LL);
      *(_QWORD *)(a2 + 32) = v8;
      if ( *(_BYTE **)(a2 + 24) != v8 )
        goto LABEL_11;
      goto LABEL_18;
    }
    sub_CB6200(a2, "liveOnEntry", 0xBu);
    v8 = *(_BYTE **)(a2 + 32);
  }
  if ( *(_BYTE **)(a2 + 24) != v8 )
  {
LABEL_11:
    *v8 = 41;
    ++*(_QWORD *)(a2 + 32);
    goto LABEL_12;
  }
LABEL_18:
  sub_CB6200(a2, (unsigned __int8 *)")", 1u);
LABEL_12:
  result = *(_QWORD *)(a1 - 32);
  if ( !result )
    return result;
  v10 = *(_DWORD *)(a1 + 84);
  if ( *(_BYTE *)result == 27 )
  {
    result = *(unsigned int *)(result + 80);
    if ( v10 != (_DWORD)result )
      return result;
  }
  else
  {
    result = *(unsigned int *)(result + 72);
    if ( v10 != (_DWORD)result )
      return result;
  }
  sub_904010(a2, "->");
  v12 = *(_QWORD *)(a1 - 32);
  if ( v12
    && (*(_BYTE *)v12 != 27 ? (v13 = *(unsigned int *)(v12 + 72)) : (v13 = *(unsigned int *)(v12 + 80)), (_DWORD)v13) )
  {
    return sub_CB59D0(a2, v13);
  }
  else
  {
    return sub_904010(a2, "liveOnEntry");
  }
}
