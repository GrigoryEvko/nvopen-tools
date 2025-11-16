// Function: sub_B096D0
// Address: 0xb096d0
//
__int64 __fastcall sub_B096D0(__int64 *a1, __int64 a2, __int64 a3, char a4, unsigned int a5, char a6)
{
  __int64 *v7; // r13
  unsigned int v8; // r12d
  char v9; // bl
  __int64 v10; // r8
  int v11; // eax
  __int64 v12; // r8
  unsigned int i; // ebx
  _QWORD *v14; // r13
  _BYTE *v15; // rax
  unsigned int v16; // r10d
  __int64 result; // rax
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // r9
  _QWORD *v24; // rcx
  __int64 v25; // [rsp+10h] [rbp-80h]
  int v26; // [rsp+18h] [rbp-78h]
  __int64 v27; // [rsp+20h] [rbp-70h]
  char v28; // [rsp+20h] [rbp-70h]
  int v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h]
  __int64 v32; // [rsp+40h] [rbp-50h] BYREF
  __int64 v33; // [rsp+48h] [rbp-48h] BYREF
  __int64 v34; // [rsp+50h] [rbp-40h]

  v7 = a1;
  v8 = a5;
  v9 = a4;
  if ( a5 )
    goto LABEL_10;
  v10 = *a1;
  v32 = a2;
  v33 = a3;
  LOBYTE(v34) = a4;
  v27 = v10;
  v29 = *(_DWORD *)(v10 + 1168);
  v30 = *(_QWORD *)(v10 + 1152);
  if ( v29 )
  {
    v11 = sub_AFB5F0(&v32, &v33);
    v12 = v27;
    v28 = v9;
    v26 = 1;
    v25 = v12;
    for ( i = (v29 - 1) & v11; ; i = v16 & (v29 - 1) )
    {
      v14 = *(_QWORD **)(v30 + 8LL * i);
      if ( v14 == (_QWORD *)-4096LL )
      {
        v7 = a1;
        v9 = v28;
        v8 = 0;
        goto LABEL_9;
      }
      if ( v14 != (_QWORD *)-8192LL )
      {
        v15 = sub_A17150((_BYTE *)v14 - 16);
        if ( v32 == *((_QWORD *)v15 + 1) )
        {
          v22 = sub_AF5140((__int64)v14, 2u);
          if ( v33 == v22 && (_BYTE)v34 == (unsigned __int8)BYTE1(*v14) >> 7 )
            break;
        }
      }
      v16 = i + v26++;
    }
    v23 = v30 + 8LL * i;
    v24 = v14;
    v9 = v28;
    v7 = a1;
    v8 = 0;
    if ( v23 != *(_QWORD *)(v25 + 1152) + 8LL * *(unsigned int *)(v25 + 1168) )
      return (__int64)v24;
  }
LABEL_9:
  result = 0;
  if ( a6 )
  {
LABEL_10:
    v18 = *v7;
    v33 = a2;
    v34 = a3;
    v19 = v18 + 1144;
    v32 = 0;
    v20 = sub_B97910(16, 3, v8);
    v21 = v20;
    if ( v20 )
      sub_AF3E50(v20, (int)v7, v8, v9, (int)&v32, 3);
    return sub_B09510(v21, v8, v19);
  }
  return result;
}
