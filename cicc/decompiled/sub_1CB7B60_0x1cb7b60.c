// Function: sub_1CB7B60
// Address: 0x1cb7b60
//
__int64 __fastcall sub_1CB7B60(__int64 a1, unsigned __int64 a2)
{
  int v2; // ebx
  unsigned int v3; // r14d
  unsigned int v5; // eax
  int v6; // edx
  unsigned int v7; // [rsp+4h] [rbp-2Ch] BYREF
  unsigned __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_1CB76C0((unsigned int *)a1, a2);
  if ( *(_BYTE *)(**(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) + 8LL) != 15 )
    return 0;
  v7 = 0;
  v3 = sub_1CB6E90(a1, a2, (__int64 *)v8, &v7);
  if ( (_BYTE)v3 )
  {
    v5 = sub_1CB76C0((unsigned int *)a1, v8[0]);
    v6 = *(_DWORD *)a1;
    if ( v5 == *(_DWORD *)a1 )
    {
      if ( v6 != v2 )
      {
LABEL_7:
        sub_1CB7560((_QWORD *)a1, a2, v6);
        return v3;
      }
    }
    else
    {
      v6 = sub_1CB71C0(a1, v7, v5);
      if ( v6 != v2 )
        goto LABEL_7;
    }
    return 0;
  }
  sub_1CB7560((_QWORD *)a1, a2, *(_DWORD *)(a1 + 4));
  return v3;
}
