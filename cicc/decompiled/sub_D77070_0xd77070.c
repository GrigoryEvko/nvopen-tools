// Function: sub_D77070
// Address: 0xd77070
//
__int64 __fastcall sub_D77070(_QWORD *a1, _QWORD *a2, _QWORD *a3, size_t a4, _QWORD *a5, size_t a6)
{
  __int64 v6; // r14
  __int64 v7; // rax
  size_t v8; // r13
  __int64 v9; // rcx
  _QWORD *v10; // r15
  __int64 v11; // rbx
  __int64 v12; // r12
  int v13; // eax
  int v14; // eax
  _DWORD *v15; // rdx
  bool v16; // zf
  __int64 result; // rax
  __int64 *v18; // rax
  __int16 v19; // [rsp+4h] [rbp-ACh]
  __int64 v20; // [rsp+8h] [rbp-A8h]
  int v21; // [rsp+1Ch] [rbp-94h] BYREF
  _QWORD v22[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v23; // [rsp+40h] [rbp-70h]
  _QWORD v24[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v25; // [rsp+70h] [rbp-40h]

  v6 = (__int64)a1;
  v7 = a1[21];
  v19 = (__int16)a2;
  v21 = 0;
  if ( *(_QWORD *)(v7 + 32) )
  {
    a4 = a6;
    a3 = a5;
  }
  v8 = a4;
  v9 = *((unsigned int *)a1 + 46);
  v10 = a3;
  if ( *((_DWORD *)a1 + 46) )
  {
    v11 = a1[22];
    v12 = 0;
    while ( 1 )
    {
      if ( v8 == *(_QWORD *)(v11 + 8) )
      {
        v20 = v9;
        if ( !v8 )
          break;
        a1 = *(_QWORD **)v11;
        a2 = v10;
        v13 = memcmp(*(const void **)v11, v10, v8);
        v9 = v20;
        if ( !v13 )
          break;
      }
      ++v12;
      v11 += 48;
      if ( v9 == v12 )
        goto LABEL_13;
    }
    v14 = *(_DWORD *)(v11 + 40);
    v21 = v14;
  }
  else
  {
LABEL_13:
    v18 = sub_CEADF0();
    a2 = v24;
    v25 = 770;
    a1 = (_QWORD *)v6;
    v23 = 1283;
    v22[0] = "Cannot find option named '";
    v24[0] = v22;
    v22[2] = v10;
    v22[3] = v8;
    v24[2] = "'!";
    result = sub_C53280(v6, (__int64)v24, 0, 0, (__int64)v18);
    if ( (_BYTE)result )
      return result;
    v14 = v21;
  }
  v15 = *(_DWORD **)(v6 + 136);
  *v15 = v14;
  v16 = *(_QWORD *)(v6 + 592) == 0;
  *(_WORD *)(v6 + 14) = v19;
  if ( v16 )
    sub_4263D6(a1, a2, v15);
  (*(void (__fastcall **)(__int64, int *, _DWORD *, __int64))(v6 + 600))(v6 + 576, &v21, v15, v9);
  return 0;
}
