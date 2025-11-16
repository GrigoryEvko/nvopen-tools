// Function: sub_2E79E40
// Address: 0x2e79e40
//
_BYTE *__fastcall sub_2E79E40(__int64 a1, __int64 a2)
{
  _BYTE *result; // rax
  void *v3; // rdx
  _BYTE *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  _BYTE *v8; // rdi
  __int64 v9; // rdx
  _BYTE *v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rbx
  __int64 *v13; // r14
  __int64 v14; // r15
  __int64 v15; // r9
  _BYTE *v16; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-98h]
  __int64 v20; // [rsp+20h] [rbp-80h]
  __int64 v21; // [rsp+28h] [rbp-78h]
  _BYTE v22[16]; // [rsp+30h] [rbp-70h] BYREF
  void (__fastcall *v23)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-60h]
  void (__fastcall *v24)(_BYTE *, __int64); // [rsp+48h] [rbp-58h]
  _QWORD v25[2]; // [rsp+50h] [rbp-50h] BYREF
  void (__fastcall *v26)(_QWORD *, _QWORD *, __int64); // [rsp+60h] [rbp-40h]
  void (__fastcall *v27)(_QWORD *, __int64); // [rsp+68h] [rbp-38h]

  result = *(_BYTE **)(a1 + 16);
  if ( *(_BYTE **)(a1 + 8) != result )
  {
    v3 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0xCu )
    {
      sub_CB6200(a2, "Jump Tables:\n", 0xDu);
    }
    else
    {
      qmemcpy(v3, "Jump Tables:\n", 13);
      *(_QWORD *)(a2 + 32) += 13LL;
    }
    v6 = (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 5;
    if ( (_DWORD)v6 )
    {
      v20 = 0;
      v18 = (unsigned int)v6;
      while ( 1 )
      {
        v7 = (unsigned int)v20;
        v8 = v22;
        sub_2E79E20((__int64)v22, v20);
        if ( !v23 )
          goto LABEL_30;
        v24(v22, a2);
        v10 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 58);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v10 + 1;
          *v10 = 58;
        }
        if ( v23 )
          v23(v22, v22, 3);
        v11 = *(_QWORD *)(a1 + 8) + 32 * v20;
        v12 = *(__int64 **)v11;
        v13 = *(__int64 **)(v11 + 8);
        if ( *(__int64 **)v11 != v13 )
          break;
LABEL_4:
        v5 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v5 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 10);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v5 + 1;
          *v5 = 10;
        }
        if ( v18 == ++v20 )
          goto LABEL_7;
      }
      while ( 1 )
      {
        v15 = *v12;
        v16 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v16 < *(_QWORD *)(a2 + 24) )
        {
          v14 = a2;
          *(_QWORD *)(a2 + 32) = v16 + 1;
          *v16 = 32;
        }
        else
        {
          v21 = *v12;
          v17 = sub_CB5D20(a2, 32);
          v15 = v21;
          v14 = v17;
        }
        v7 = v15;
        v8 = v25;
        sub_2E31000(v25, v15);
        if ( !v26 )
          break;
        v27(v25, v14);
        if ( v26 )
          v26(v25, v25, 3);
        if ( v13 == ++v12 )
          goto LABEL_4;
      }
LABEL_30:
      sub_4263D6(v8, v7, v9);
    }
LABEL_7:
    result = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    {
      return (_BYTE *)sub_CB5D20(a2, 10);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = result + 1;
      *result = 10;
    }
  }
  return result;
}
