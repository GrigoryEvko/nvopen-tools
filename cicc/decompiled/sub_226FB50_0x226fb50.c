// Function: sub_226FB50
// Address: 0x226fb50
//
__int64 __fastcall sub_226FB50(_QWORD *a1, _QWORD *a2, _QWORD *a3, size_t a4, _QWORD *a5, size_t a6)
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
  bool v15; // zf
  __int64 result; // rax
  __int64 *v17; // rax
  __int16 v18; // [rsp+4h] [rbp-ACh]
  __int64 v19; // [rsp+8h] [rbp-A8h]
  int v20; // [rsp+1Ch] [rbp-94h] BYREF
  _QWORD v21[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v22; // [rsp+40h] [rbp-70h]
  _QWORD v23[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v24; // [rsp+70h] [rbp-40h]

  v6 = (__int64)a1;
  v7 = a1[21];
  v18 = (__int16)a2;
  v20 = 0;
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
        v19 = v9;
        if ( !v8 )
          break;
        a1 = *(_QWORD **)v11;
        a2 = v10;
        v13 = memcmp(*(const void **)v11, v10, v8);
        v9 = v19;
        if ( !v13 )
          break;
      }
      ++v12;
      v11 += 48;
      if ( v9 == v12 )
        goto LABEL_13;
    }
    v14 = *(_DWORD *)(v11 + 40);
    v20 = v14;
  }
  else
  {
LABEL_13:
    v17 = sub_CEADF0();
    a2 = v23;
    v24 = 770;
    a1 = (_QWORD *)v6;
    v22 = 1283;
    v21[0] = "Cannot find option named '";
    v23[0] = v21;
    v21[2] = v10;
    v21[3] = v8;
    v23[2] = "'!";
    result = sub_C53280(v6, (__int64)v23, 0, 0, (__int64)v17);
    if ( (_BYTE)result )
      return result;
    v14 = v20;
  }
  *(_DWORD *)(v6 + 136) = v14;
  v15 = *(_QWORD *)(v6 + 592) == 0;
  *(_WORD *)(v6 + 14) = v18;
  if ( v15 )
    sub_4263D6(a1, a2, a3);
  (*(void (__fastcall **)(__int64, int *, _QWORD *, __int64))(v6 + 600))(v6 + 576, &v20, a3, v9);
  return 0;
}
