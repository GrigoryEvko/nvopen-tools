// Function: sub_398A890
// Address: 0x398a890
//
void __fastcall sub_398A890(__int64 a1, void (__fastcall ***a2)(_QWORD, _QWORD, _QWORD), __int64 a3)
{
  __int64 v5; // r8
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned __int8 *v12; // rbx
  _BYTE *v13; // r9
  __int64 v14; // r11
  size_t v15; // r12
  _QWORD *v16; // rax
  unsigned __int8 v17; // cl
  void (__fastcall *v18)(_QWORD, _QWORD, _QWORD); // r10
  __int64 v19; // rax
  _QWORD *v20; // rdi
  _BYTE *src; // [rsp+8h] [rbp-A8h]
  __int64 v22; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v23; // [rsp+18h] [rbp-98h]
  __int64 v24; // [rsp+18h] [rbp-98h]
  unsigned __int8 v25; // [rsp+20h] [rbp-90h]
  void (__fastcall *v26)(_QWORD, _QWORD, _QWORD); // [rsp+20h] [rbp-90h]
  unsigned __int8 v27; // [rsp+20h] [rbp-90h]
  void (__fastcall *v28)(_QWORD, _QWORD, _QWORD); // [rsp+28h] [rbp-88h]
  void (__fastcall *v29)(_QWORD, _QWORD, _QWORD); // [rsp+28h] [rbp-88h]
  unsigned __int8 *v30; // [rsp+30h] [rbp-80h]
  __int64 v31; // [rsp+38h] [rbp-78h]
  unsigned __int64 v32[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v33; // [rsp+50h] [rbp-60h]
  __int64 v34[2]; // [rsp+60h] [rbp-50h] BYREF
  _QWORD v35[8]; // [rsp+70h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a1 + 1336);
  v6 = *(_QWORD *)(a3 + 24);
  v7 = *(_QWORD *)(a3 + 16);
  v8 = a3 - v5;
  v9 = *(_QWORD *)(a1 + 2648) + 32 * v6;
  if ( (v8 >> 5) + 1 == *(_DWORD *)(a1 + 1344) )
  {
    v31 = v9 + 32 * (*(unsigned int *)(a1 + 2656) - v6);
    v11 = *(unsigned int *)(a1 + 2384);
  }
  else
  {
    v10 = v5 + v8 + 32;
    v31 = v9 + 32 * (*(_QWORD *)(v10 + 24) - v6);
    v11 = *(_QWORD *)(v10 + 16);
  }
  v12 = (unsigned __int8 *)(v7 + *(_QWORD *)(a1 + 2376));
  v30 = (unsigned __int8 *)(*(_QWORD *)(a1 + 2376) + v11);
  if ( v30 != v12 )
  {
    while ( 1 )
    {
      v17 = *v12;
      v18 = **a2;
      if ( v31 == v9 )
      {
        v34[0] = (__int64)v35;
        v25 = v17;
        v28 = v18;
        sub_3984920(v34, byte_3F871B3, (__int64)byte_3F871B3);
        v17 = v25;
        v18 = v28;
        goto LABEL_11;
      }
      v34[0] = (__int64)v35;
      v13 = *(_BYTE **)v9;
      v14 = v9 + 32;
      v15 = *(_QWORD *)(v9 + 8);
      if ( &v13[v15] && !v13 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v32[0] = v15;
      if ( v15 > 0xF )
        break;
      if ( v15 == 1 )
      {
        LOBYTE(v35[0]) = *v13;
        v16 = v35;
      }
      else
      {
        if ( v15 )
        {
          v20 = v35;
          goto LABEL_19;
        }
        v16 = v35;
      }
LABEL_10:
      v34[1] = v15;
      *((_BYTE *)v16 + v15) = 0;
      v9 = v14;
LABEL_11:
      v32[0] = (unsigned __int64)v34;
      v33 = 260;
      v18(a2, v17, v32);
      if ( (_QWORD *)v34[0] != v35 )
        j_j___libc_free_0(v34[0]);
      if ( v30 == ++v12 )
        return;
    }
    src = v13;
    v22 = v14;
    v23 = v17;
    v26 = v18;
    v19 = sub_22409D0((__int64)v34, v32, 0);
    v18 = v26;
    v34[0] = v19;
    v20 = (_QWORD *)v19;
    v17 = v23;
    v14 = v22;
    v35[0] = v32[0];
    v13 = src;
LABEL_19:
    v24 = v14;
    v27 = v17;
    v29 = v18;
    memcpy(v20, v13, v15);
    v15 = v32[0];
    v16 = (_QWORD *)v34[0];
    v18 = v29;
    v17 = v27;
    v14 = v24;
    goto LABEL_10;
  }
}
