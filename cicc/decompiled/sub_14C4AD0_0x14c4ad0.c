// Function: sub_14C4AD0
// Address: 0x14c4ad0
//
__int64 __fastcall sub_14C4AD0(__int64 a1, __int64 *a2, int a3)
{
  __int64 v3; // rax
  bool v4; // sf
  unsigned int v5; // r13d
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rdi
  __int64 v11; // rsi
  bool v12; // al
  __int64 v16; // [rsp+20h] [rbp-A0h]
  unsigned int *v17; // [rsp+28h] [rbp-98h]
  unsigned __int64 v19[2]; // [rsp+40h] [rbp-80h] BYREF
  _BYTE v20[112]; // [rsp+50h] [rbp-70h] BYREF

  v3 = *a2;
  v19[0] = (unsigned __int64)v20;
  v4 = *(__int16 *)(v3 + 18) < 0;
  v16 = v3;
  v19[1] = 0x400000000LL;
  if ( v4 )
    sub_161F980(v3, v19);
  v5 = 1;
  v17 = (unsigned int *)&unk_428FDC0;
  v6 = v16;
  v7 = *(_QWORD *)(v16 + 48);
  if ( !v7 )
    goto LABEL_15;
LABEL_4:
  v7 = sub_1625790(v16, v5);
  if ( a3 != 1 && v7 )
  {
    v8 = 2;
    v9 = v7;
    do
    {
      v10 = a2[v8 - 1];
      v11 = *(_QWORD *)(v10 + 48);
      if ( v11 || *(__int16 *)(v10 + 18) < 0 )
        v11 = sub_1625790(v10, v5);
      switch ( v5 )
      {
        case 0u:
        case 2u:
        case 4u:
        case 5u:
        case 6u:
        case 8u:
        case 9u:
          v9 = sub_1630FC0(v9, v11);
          break;
        case 1u:
          v9 = sub_14A8140(v9, v11);
          break;
        case 3u:
          v9 = sub_161F2A0(v9, v11);
          break;
        case 7u:
          v9 = sub_1631A90(v9, v11);
          break;
      }
      v12 = a3 == (_DWORD)v8++;
    }
    while ( !v12 && v9 != 0 );
    v7 = v9;
  }
  while ( 1 )
  {
    sub_1625C10(a1, v5, v7);
    if ( ++v17 == (unsigned int *)jpt_14CF649 )
      break;
    v5 = *v17;
    v6 = v16;
    v7 = *(_QWORD *)(v16 + 48);
    if ( v7 )
      goto LABEL_4;
LABEL_15:
    if ( *(__int16 *)(v6 + 18) < 0 )
      goto LABEL_4;
  }
  if ( (_BYTE *)v19[0] != v20 )
    _libc_free(v19[0]);
  return a1;
}
