// Function: sub_BD0B90
// Address: 0xbd0b90
//
_QWORD *__fastcall sub_BD0B90(_QWORD *a1, _QWORD *a2, __int64 a3, char a4)
{
  _QWORD *v4; // r8
  __int64 v8; // rbx
  unsigned int v9; // ecx
  __int64 v10; // r15
  unsigned int v11; // esi
  __int64 v12; // r9
  int v13; // eax
  __int64 v14; // r15
  int v15; // eax
  unsigned int v16; // ecx
  unsigned int v17; // r10d
  __int64 v18; // rdx
  int i; // r11d
  __int64 v20; // rax
  int v21; // eax
  _QWORD *v22; // rax
  _QWORD *v23; // r15
  __int64 v24; // rax
  int v25; // eax
  size_t v26; // rax
  int v28; // eax
  __int64 v29; // [rsp+0h] [rbp-90h]
  _QWORD *v30; // [rsp+8h] [rbp-88h]
  int v31; // [rsp+10h] [rbp-80h]
  unsigned int v32; // [rsp+14h] [rbp-7Ch]
  _QWORD *v33; // [rsp+20h] [rbp-70h]
  unsigned int v34; // [rsp+20h] [rbp-70h]
  _QWORD *v35; // [rsp+28h] [rbp-68h]
  unsigned int v36; // [rsp+28h] [rbp-68h]
  __int64 v37; // [rsp+28h] [rbp-68h]
  _QWORD *v38; // [rsp+28h] [rbp-68h]
  unsigned __int64 v39; // [rsp+38h] [rbp-58h] BYREF
  void *s1; // [rsp+40h] [rbp-50h] BYREF
  __int64 v41; // [rsp+48h] [rbp-48h]
  char v42[64]; // [rsp+50h] [rbp-40h] BYREF

  v4 = a1;
  v8 = *a1;
  v42[0] = a4;
  s1 = a2;
  v9 = *(_DWORD *)(v8 + 2960);
  v41 = a3;
  if ( !v9 )
  {
    v39 = 0;
    v10 = v8 + 2936;
    ++*(_QWORD *)(v8 + 2936);
LABEL_3:
    v35 = v4;
    v11 = 2 * v9;
    goto LABEL_4;
  }
  v14 = *(_QWORD *)(v8 + 2944);
  v36 = v9;
  v39 = sub_BCC330(a2, (__int64)&a2[a3]);
  v15 = sub_BCC160((__int64 *)&v39, v42);
  v4 = a1;
  v16 = v36 - 1;
  v17 = (v36 - 1) & v15;
  v12 = v14 + 8LL * v17;
  v18 = *(_QWORD *)v12;
  if ( *(_QWORD *)v12 == -4096 )
  {
    v11 = *(_DWORD *)(v8 + 2960);
LABEL_13:
    v9 = v11;
    goto LABEL_14;
  }
  v37 = 0;
  for ( i = 1; ; ++i )
  {
    if ( v18 == -8192 )
    {
      if ( v37 )
        v12 = v37;
      v37 = v12;
    }
    else if ( ((*(_DWORD *)(v18 + 8) & 0x200) != 0) == v42[0] )
    {
      v20 = *(unsigned int *)(v18 + 12);
      if ( v20 == v41 )
      {
        v26 = 8 * v20;
        v30 = v4;
        v31 = i;
        v32 = v16;
        v34 = v17;
        if ( !v26 )
          return *(_QWORD **)v12;
        v29 = v12;
        v28 = memcmp(s1, *(const void **)(v18 + 16), v26);
        v12 = v29;
        v17 = v34;
        v16 = v32;
        i = v31;
        v4 = v30;
        if ( !v28 )
          return *(_QWORD **)v12;
      }
    }
    v17 = v16 & (i + v17);
    v12 = v14 + 8LL * v17;
    v18 = *(_QWORD *)v12;
    if ( *(_QWORD *)v12 == -4096 )
      break;
  }
  v11 = *(_DWORD *)(v8 + 2960);
  v9 = v11;
  if ( !v37 )
    goto LABEL_13;
  v12 = v37;
LABEL_14:
  v39 = v12;
  v21 = *(_DWORD *)(v8 + 2952);
  v10 = v8 + 2936;
  ++*(_QWORD *)(v8 + 2936);
  v13 = v21 + 1;
  if ( 4 * v13 >= 3 * v11 )
    goto LABEL_3;
  if ( v11 - (v13 + *(_DWORD *)(v8 + 2956)) > v11 >> 3 )
    goto LABEL_16;
  v35 = v4;
LABEL_4:
  sub_BCF650(v10, v11);
  sub_BCC910(v10, (__int64)&s1, &v39);
  v12 = v39;
  v4 = v35;
  v13 = *(_DWORD *)(v8 + 2952) + 1;
LABEL_16:
  *(_DWORD *)(v8 + 2952) = v13;
  if ( *(_QWORD *)v12 != -4096 )
    --*(_DWORD *)(v8 + 2956);
  *(_QWORD *)v12 = 0;
  v38 = (_QWORD *)v12;
  v33 = v4;
  v22 = (_QWORD *)sub_A777F0(0x20u, (__int64 *)(*v4 + 2640LL));
  v23 = v22;
  if ( v22 )
  {
    *v22 = v33;
    v24 = v22[1];
    v23[2] = 0;
    v23[3] = 0;
    v23[1] = (unsigned int)v24 & 0xFFFFFF00 | 0xFLL;
  }
  v25 = *((unsigned __int8 *)v23 + 8);
  BYTE1(v25) = 4;
  *((_DWORD *)v23 + 2) = v25;
  sub_BD0B50((__int64)v23, a2, a3, a4);
  *v38 = v23;
  return v23;
}
