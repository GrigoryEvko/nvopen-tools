// Function: sub_15D2530
// Address: 0x15d2530
//
__int64 __fastcall sub_15D2530(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        int a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // r14
  unsigned int v12; // eax
  char *v13; // rdx
  __int64 *v14; // rax
  char **v15; // r12
  char **v16; // r14
  _QWORD *v17; // rsi
  __int64 v18; // r8
  char *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // r8
  unsigned int v22; // edx
  __int64 v23; // rdi
  char *v24; // r10
  int v26; // edi
  int v27; // ecx
  __int64 v28; // [rsp+28h] [rbp-2D8h]
  __int64 *v29; // [rsp+38h] [rbp-2C8h]
  __int64 v31; // [rsp+48h] [rbp-2B8h] BYREF
  char *v32; // [rsp+58h] [rbp-2A8h] BYREF
  char *v33; // [rsp+60h] [rbp-2A0h] BYREF
  __int64 v34; // [rsp+68h] [rbp-298h] BYREF
  char **v35; // [rsp+70h] [rbp-290h] BYREF
  int v36; // [rsp+78h] [rbp-288h]
  char v37; // [rsp+80h] [rbp-280h] BYREF
  _QWORD *v38; // [rsp+C0h] [rbp-240h] BYREF
  __int64 v39; // [rsp+C8h] [rbp-238h]
  _QWORD v40[70]; // [rsp+D0h] [rbp-230h] BYREF

  v9 = a1 + 24;
  v31 = a2;
  v38 = v40;
  v40[0] = a2;
  v34 = a2;
  v39 = 0x4000000001LL;
  if ( (unsigned __int8)sub_15CE630(a1 + 24, &v34, &v35) )
    *((_DWORD *)sub_15D1D60(v9, &v31) + 3) = a4;
  v12 = v39;
  if ( (_DWORD)v39 )
  {
LABEL_6:
    while ( 1 )
    {
      v13 = (char *)v38[v12 - 1];
      LODWORD(v39) = v12 - 1;
      v32 = v13;
      v14 = sub_15D1D60(v9, (__int64 *)&v32);
      if ( !*((_DWORD *)v14 + 2) )
        break;
LABEL_5:
      v12 = v39;
      if ( !(_DWORD)v39 )
        goto LABEL_22;
    }
    ++a3;
    v14[3] = (__int64)v32;
    *((_DWORD *)v14 + 4) = a3;
    *((_DWORD *)v14 + 2) = a3;
    sub_15CE600(a1, &v32);
    sub_15CF6C0((__int64)&v35, (__int64)v32, *(_QWORD *)(a1 + 56));
    v15 = &v35[v36];
    if ( v35 == v15 )
      goto LABEL_20;
    v28 = v9;
    v16 = v35;
    while ( 1 )
    {
      v19 = *v16;
      v20 = *(unsigned int *)(a1 + 48);
      v33 = *v16;
      if ( !(_DWORD)v20 )
        goto LABEL_9;
      v21 = *(_QWORD *)(a1 + 32);
      v22 = (v20 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v23 = v21 + 72LL * v22;
      v24 = *(char **)v23;
      if ( v19 != *(char **)v23 )
        break;
LABEL_15:
      if ( v23 == v21 + 72 * v20 || !*(_DWORD *)(v23 + 8) )
        goto LABEL_9;
      if ( v19 == v32 )
      {
LABEL_12:
        if ( v15 == ++v16 )
          goto LABEL_19;
      }
      else
      {
        ++v16;
        sub_15CDD90(v23 + 40, &v32);
        if ( v15 == v16 )
        {
LABEL_19:
          v9 = v28;
          v15 = v35;
LABEL_20:
          if ( v15 == (char **)&v37 )
            goto LABEL_5;
          _libc_free((unsigned __int64)v15);
          v12 = v39;
          if ( !(_DWORD)v39 )
            goto LABEL_22;
          goto LABEL_6;
        }
      }
    }
    v26 = 1;
    while ( v24 != (char *)-8LL )
    {
      v27 = v26 + 1;
      v22 = (v20 - 1) & (v26 + v22);
      v23 = v21 + 72LL * v22;
      v24 = *(char **)v23;
      if ( v19 == *(char **)v23 )
        goto LABEL_15;
      v26 = v27;
    }
LABEL_9:
    v34 = (__int64)v19;
    if ( *(_DWORD *)(sub_15CC510(a9, (__int64)v19) + 16) > a7 )
    {
      v29 = sub_15D1D60(v28, (__int64 *)&v33);
      sub_15CDD90((__int64)&v38, &v33);
      *((_DWORD *)v29 + 3) = a3;
      sub_15CDD90((__int64)(v29 + 5), &v32);
    }
    else
    {
      v17 = (_QWORD *)(*(_QWORD *)a8 + 8LL * *(unsigned int *)(a8 + 8));
      if ( v17 == sub_15CBCA0(*(_QWORD **)a8, (__int64)v17, &v34) )
        sub_15CDD90(v18, &v34);
    }
    goto LABEL_12;
  }
LABEL_22:
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
  return a3;
}
