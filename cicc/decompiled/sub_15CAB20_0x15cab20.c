// Function: sub_15CAB20
// Address: 0x15cab20
//
__int64 __fastcall sub_15CAB20(__int64 a1, _BYTE *a2, size_t a3)
{
  unsigned int v5; // eax
  __int64 v6; // rbx
  size_t v7; // r15
  _BYTE *v8; // rdi
  __int64 result; // rax
  __int64 v10; // rax
  size_t v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = *(_DWORD *)(a1 + 96);
  if ( v5 >= *(_DWORD *)(a1 + 100) )
  {
    sub_14B3F20(a1 + 88, 0);
    v5 = *(_DWORD *)(a1 + 96);
  }
  v6 = *(_QWORD *)(a1 + 88) + 88LL * v5;
  if ( v6 )
  {
    v7 = a3;
    *(_QWORD *)v6 = v6 + 16;
    sub_15C7EA0((__int64 *)v6, "String", (__int64)"");
    v8 = (_BYTE *)(v6 + 48);
    *(_QWORD *)(v6 + 32) = v6 + 48;
    if ( !a2 )
    {
      *(_QWORD *)(v6 + 40) = 0;
      *(_BYTE *)(v6 + 48) = 0;
      goto LABEL_10;
    }
    v11[0] = a3;
    if ( a3 > 0xF )
    {
      v10 = sub_22409D0(v6 + 32, v11, 0);
      *(_QWORD *)(v6 + 32) = v10;
      v8 = (_BYTE *)v10;
      *(_QWORD *)(v6 + 48) = v11[0];
    }
    else
    {
      if ( a3 == 1 )
      {
        *(_BYTE *)(v6 + 48) = *a2;
LABEL_8:
        *(_QWORD *)(v6 + 40) = v7;
        v8[v7] = 0;
LABEL_10:
        *(_QWORD *)(v6 + 64) = 0;
        *(_QWORD *)(v6 + 72) = 0;
        *(_QWORD *)(v6 + 80) = 0;
        v5 = *(_DWORD *)(a1 + 96);
        goto LABEL_11;
      }
      if ( !a3 )
        goto LABEL_8;
    }
    memcpy(v8, a2, a3);
    v7 = v11[0];
    v8 = *(_BYTE **)(v6 + 32);
    goto LABEL_8;
  }
LABEL_11:
  result = v5 + 1;
  *(_DWORD *)(a1 + 96) = result;
  return result;
}
