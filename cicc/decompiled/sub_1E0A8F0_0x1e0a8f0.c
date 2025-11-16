// Function: sub_1E0A8F0
// Address: 0x1e0a8f0
//
_BYTE *__fastcall sub_1E0A8F0(__int64 a1, __int64 a2)
{
  _BYTE *result; // rax
  void *v5; // rdx
  __int64 v6; // rsi
  _BYTE *v7; // rdi
  __int64 v8; // rdx
  _WORD *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r14
  _BYTE *v13; // rax
  int v14; // [rsp-90h] [rbp-90h]
  unsigned int v15; // [rsp-8Ch] [rbp-8Ch]
  __int64 v16; // [rsp-88h] [rbp-88h]
  __int64 v17; // [rsp-80h] [rbp-80h]
  _BYTE v18[16]; // [rsp-78h] [rbp-78h] BYREF
  void (__fastcall *v19)(_BYTE *, _BYTE *, __int64); // [rsp-68h] [rbp-68h]
  void (__fastcall *v20)(_BYTE *, __int64); // [rsp-60h] [rbp-60h]
  _QWORD v21[2]; // [rsp-58h] [rbp-58h] BYREF
  void (__fastcall *v22)(_QWORD *, _QWORD *, __int64); // [rsp-48h] [rbp-48h]
  void (__fastcall *v23)(_QWORD *, __int64); // [rsp-40h] [rbp-40h]

  result = *(_BYTE **)(a1 + 16);
  if ( *(_BYTE **)(a1 + 8) != result )
  {
    v5 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v5 <= 0xCu )
    {
      sub_16E7EE0(a2, "Jump Tables:\n", 0xDu);
    }
    else
    {
      qmemcpy(v5, "Jump Tables:\n", 13);
      *(_QWORD *)(a2 + 24) += 13LL;
    }
    v14 = -1431655765 * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3);
    if ( v14 )
    {
      v17 = 0;
      v15 = 0;
      while ( 1 )
      {
        v6 = v15;
        v7 = v18;
        sub_1E0A8D0((__int64)v18, v15);
        if ( !v19 )
          goto LABEL_27;
        v20(v18, a2);
        v9 = *(_WORD **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v9 <= 1u )
        {
          sub_16E7EE0(a2, ": ", 2u);
        }
        else
        {
          *v9 = 8250;
          *(_QWORD *)(a2 + 24) += 2LL;
        }
        if ( v19 )
          v19(v18, v18, 3);
        v10 = (__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + v17 + 8) - *(_QWORD *)(*(_QWORD *)(a1 + 8) + v17)) >> 3;
        if ( (_DWORD)v10 )
          break;
LABEL_21:
        ++v15;
        v17 += 24;
        if ( v14 == v15 )
          goto LABEL_22;
      }
      v11 = 0;
      v16 = 8LL * (unsigned int)v10;
      while ( 1 )
      {
        v13 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v13 < *(_QWORD *)(a2 + 16) )
        {
          v12 = a2;
          *(_QWORD *)(a2 + 24) = v13 + 1;
          *v13 = 32;
        }
        else
        {
          v12 = sub_16E7DE0(a2, 32);
        }
        v7 = v21;
        v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + v17) + v11);
        sub_1DD5B60(v21, v6);
        if ( !v22 )
          break;
        v23(v21, v12);
        if ( v22 )
          v22(v21, v21, 3);
        v11 += 8;
        if ( v11 == v16 )
          goto LABEL_21;
      }
LABEL_27:
      sub_4263D6(v7, v6, v8);
    }
LABEL_22:
    result = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 16) )
    {
      return (_BYTE *)sub_16E7DE0(a2, 10);
    }
    else
    {
      *(_QWORD *)(a2 + 24) = result + 1;
      *result = 10;
    }
  }
  return result;
}
