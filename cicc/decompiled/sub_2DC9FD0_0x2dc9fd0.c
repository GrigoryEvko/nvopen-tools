// Function: sub_2DC9FD0
// Address: 0x2dc9fd0
//
__int16 __fastcall sub_2DC9FD0(_QWORD *a1)
{
  __int64 v2; // r14
  __int64 *v3; // rdi
  __int64 v4; // rax
  __int64 (*v5)(void); // rdx
  __int64 (*v6)(void); // rax
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r12
  int v11; // eax
  bool v12; // r12
  char v13; // bl
  __int16 result; // ax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // [rsp+10h] [rbp-50h]
  char v18; // [rsp+1Eh] [rbp-42h]
  bool v19; // [rsp+1Fh] [rbp-41h]
  __int64 v20; // [rsp+20h] [rbp-40h]
  _QWORD *v21; // [rsp+28h] [rbp-38h]

  v2 = 0;
  v3 = (__int64 *)a1[2];
  v4 = *v3;
  v5 = *(__int64 (**)(void))(*v3 + 128);
  if ( v5 != sub_2DAC790 )
  {
    v2 = v5();
    v4 = *(_QWORD *)a1[2];
  }
  v6 = *(__int64 (**)(void))(v4 + 144);
  v17 = 0;
  if ( v6 != sub_2C8F680 )
    v17 = v6();
  v21 = (_QWORD *)a1[41];
  if ( v21 == a1 + 40 )
  {
    v13 = 1;
    v12 = 0;
  }
  else
  {
    v18 = 1;
    v19 = 0;
    do
    {
      v20 = (__int64)v21;
      v7 = v21[7];
      v8 = (__int64)(v21 + 6);
      while ( v8 != v7 )
      {
        while ( 1 )
        {
          if ( !v7 )
            BUG();
          v9 = v7;
          if ( (*(_BYTE *)v7 & 4) == 0 && (*(_BYTE *)(v7 + 44) & 8) != 0 )
          {
            do
              v9 = *(_QWORD *)(v9 + 8);
            while ( (*(_BYTE *)(v9 + 44) & 8) != 0 );
          }
          v10 = *(_QWORD *)(v9 + 8);
          v11 = *(unsigned __int16 *)(v7 + 68);
          if ( v11 == *(_DWORD *)(v2 + 64) || v11 == *(_DWORD *)(v2 + 68) || (unsigned __int8)sub_2E89070(v7) )
            *(_BYTE *)(a1[6] + 65LL) = 1;
          if ( (*(_QWORD *)(*(_QWORD *)(v7 + 16) + 24LL) & 0x8000000LL) != 0 )
            break;
          v7 = v10;
          if ( v8 == v10 )
            goto LABEL_19;
        }
        v15 = v7;
        v19 = (*(_QWORD *)(*(_QWORD *)(v7 + 16) + 24LL) & 0x8000000LL) != 0;
        v7 = v10;
        v16 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *))(*(_QWORD *)v17 + 2600LL))(v17, v15, v21);
        if ( v21 != (_QWORD *)v16 )
        {
          v20 = v16;
          v7 = *(_QWORD *)(v16 + 56);
          v8 = v16 + 48;
          v21 = (_QWORD *)v16;
          v18 = 0;
        }
      }
LABEL_19:
      v21 = *(_QWORD **)(v20 + 8);
    }
    while ( a1 + 40 != v21 );
    v12 = v19;
    v13 = v18;
  }
  (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v17 + 1800LL))(v17, a1);
  LOBYTE(result) = v12;
  HIBYTE(result) = v13;
  return result;
}
