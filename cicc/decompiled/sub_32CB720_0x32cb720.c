// Function: sub_32CB720
// Address: 0x32cb720
//
__int64 __fastcall sub_32CB720(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  unsigned int v6; // ebx
  unsigned __int16 *v7; // rax
  int v8; // r15d
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r11
  __int64 result; // rax
  __int64 v14; // rdi
  int v15; // r9d
  __int64 v16; // r14
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // rsi
  __int128 *v20; // rbx
  int v21; // [rsp+10h] [rbp-90h]
  __int64 v22; // [rsp+10h] [rbp-90h]
  __int64 v23; // [rsp+20h] [rbp-80h]
  int v24; // [rsp+28h] [rbp-78h]
  __int64 v25; // [rsp+28h] [rbp-78h]
  __int64 v26; // [rsp+30h] [rbp-70h] BYREF
  int v27; // [rsp+38h] [rbp-68h]
  __int64 v28; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v29; // [rsp+48h] [rbp-58h]
  __int64 v30; // [rsp+50h] [rbp-50h] BYREF
  int v31; // [rsp+58h] [rbp-48h]
  __int64 v32; // [rsp+60h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(v4 + 8);
  v7 = *(unsigned __int16 **)(a2 + 48);
  v23 = v5;
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v31 = *(_DWORD *)(a2 + 28);
  v24 = v9;
  v10 = *(_QWORD *)(*a1 + 1024);
  v30 = *a1;
  v32 = v10;
  *(_QWORD *)(v30 + 1024) = &v30;
  v11 = *(_QWORD *)(a2 + 80);
  v28 = v5;
  v12 = *a1;
  v29 = v6;
  v26 = v11;
  if ( v11 )
  {
    v21 = v12;
    sub_B96E90((__int64)&v26, v11, 1);
    LODWORD(v12) = v21;
  }
  v27 = *(_DWORD *)(a2 + 72);
  result = sub_3402EA0(v12, 244, (unsigned int)&v26, v8, v9, 0, (__int64)&v28, 1);
  if ( v26 )
  {
    v22 = result;
    sub_B91220((__int64)&v26, v26);
    result = v22;
  }
  if ( !result )
  {
    v14 = a1[1];
    LODWORD(v28) = 2;
    result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD, _QWORD, __int64 *))(*(_QWORD *)v14 + 2264LL))(
               v14,
               v23,
               v6,
               *a1,
               *((unsigned __int8 *)a1 + 33),
               *((unsigned __int8 *)a1 + 35),
               &v28);
    if ( !result )
    {
      if ( *(_DWORD *)(v23 + 24) == 97
        && ((v16 = *a1, (*(_BYTE *)(*(_QWORD *)*a1 + 864LL) & 0x10) != 0) || *(char *)(a2 + 28) < 0)
        && (v17 = *(_QWORD *)(v23 + 56)) != 0 )
      {
        v18 = 1;
        do
        {
          if ( *(_DWORD *)(v17 + 8) == v6 )
          {
            if ( !v18 )
              goto LABEL_20;
            v17 = *(_QWORD *)(v17 + 32);
            if ( !v17 )
              goto LABEL_21;
            if ( *(_DWORD *)(v17 + 8) == v6 )
              goto LABEL_20;
            v18 = 0;
          }
          v17 = *(_QWORD *)(v17 + 32);
        }
        while ( v17 );
        if ( v18 == 1 )
          goto LABEL_20;
LABEL_21:
        v19 = *(_QWORD *)(a2 + 80);
        v20 = *(__int128 **)(v23 + 40);
        v28 = v19;
        if ( v19 )
          sub_B96E90((__int64)&v28, v19, 1);
        v29 = *(_DWORD *)(a2 + 72);
        result = sub_3406EB0(v16, 97, (unsigned int)&v28, v8, v24, v15, *(__int128 *)((char *)v20 + 40), *v20);
        if ( v28 )
        {
          v25 = result;
          sub_B91220((__int64)&v28, v28);
          result = v25;
        }
      }
      else
      {
LABEL_20:
        result = sub_32CAE50(a1, a2);
      }
    }
  }
  *(_QWORD *)(v30 + 1024) = v32;
  return result;
}
