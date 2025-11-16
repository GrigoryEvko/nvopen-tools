// Function: sub_B046D0
// Address: 0xb046d0
//
__int64 __fastcall sub_B046D0(__int64 *a1, __int64 a2, unsigned __int8 a3, __int64 a4, unsigned int a5, char a6)
{
  int v7; // r13d
  unsigned int v9; // edx
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  int v13; // ebx
  int v14; // eax
  int v15; // r11d
  __int64 v16; // r8
  int v17; // ecx
  unsigned int v18; // r10d
  __int64 *v19; // r9
  __int64 v20; // rbx
  unsigned int v21; // r10d
  __int64 result; // rax
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // rdi
  unsigned __int8 v28; // r8
  char v29; // al
  __int64 v30; // rax
  __int64 *v31; // [rsp+0h] [rbp-C0h]
  __int64 *v32; // [rsp+0h] [rbp-C0h]
  int v33; // [rsp+Ch] [rbp-B4h]
  int v34; // [rsp+Ch] [rbp-B4h]
  unsigned int v35; // [rsp+10h] [rbp-B0h]
  unsigned int v36; // [rsp+10h] [rbp-B0h]
  int v37; // [rsp+14h] [rbp-ACh]
  int v38; // [rsp+14h] [rbp-ACh]
  __int64 v39; // [rsp+18h] [rbp-A8h]
  __int64 v40; // [rsp+20h] [rbp-A0h]
  unsigned int v41; // [rsp+28h] [rbp-98h]
  unsigned int v42; // [rsp+28h] [rbp-98h]
  unsigned __int8 v43; // [rsp+2Fh] [rbp-91h]
  __int64 v44; // [rsp+38h] [rbp-88h]
  __int64 v45; // [rsp+38h] [rbp-88h]
  __int64 v46; // [rsp+38h] [rbp-88h]
  __int64 v48; // [rsp+50h] [rbp-70h]
  __int64 v49; // [rsp+50h] [rbp-70h]
  __int64 v50; // [rsp+50h] [rbp-70h]
  __int64 v52; // [rsp+58h] [rbp-68h]
  __int64 v53; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v54; // [rsp+68h] [rbp-58h]
  __int64 v55; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v56; // [rsp+78h] [rbp-48h]
  __int64 v57; // [rsp+80h] [rbp-40h] BYREF
  unsigned __int8 v58; // [rsp+88h] [rbp-38h]

  v7 = (int)a1;
  if ( a5 )
    goto LABEL_19;
  v9 = *(_DWORD *)(a2 + 8);
  v54 = v9;
  if ( v9 > 0x40 )
  {
    sub_C43780(&v53, a2);
    v9 = v54;
    v10 = v53;
  }
  else
  {
    v10 = *(_QWORD *)a2;
    v53 = *(_QWORD *)a2;
  }
  v55 = v10;
  v11 = *a1;
  v56 = v9;
  v57 = a4;
  v58 = a3;
  v12 = *(_QWORD *)(v11 + 864);
  v13 = *(_DWORD *)(v11 + 880);
  v54 = 0;
  v48 = v12;
  v44 = v11;
  if ( v13 )
  {
    v14 = sub_AFB7E0((__int64)&v55, &v57);
    v15 = v13 - 1;
    v9 = v56;
    v16 = v44;
    v17 = 1;
    v18 = (v13 - 1) & v14;
    v40 = v55;
    v43 = v58;
    v39 = v57;
    while ( 1 )
    {
      v19 = (__int64 *)(v48 + 8LL * v18);
      v20 = *v19;
      if ( *v19 == -4096 )
        break;
      if ( v20 != -8192 && v9 == *(_DWORD *)(v20 + 24) )
      {
        if ( v9 <= 0x40 )
        {
          if ( v40 != *(_QWORD *)(v20 + 16) )
            goto LABEL_9;
        }
        else
        {
          v31 = (__int64 *)(v48 + 8LL * v18);
          v33 = v17;
          v35 = v18;
          v37 = v15;
          v41 = v9;
          v45 = v16;
          v29 = sub_C43C50(&v55, v20 + 16);
          v16 = v45;
          v9 = v41;
          v15 = v37;
          v18 = v35;
          v17 = v33;
          v19 = v31;
          if ( !v29 )
            goto LABEL_9;
        }
        if ( v43 == (*(_DWORD *)(v20 + 4) != 0) )
        {
          v32 = v19;
          v34 = v17;
          v36 = v18;
          v38 = v15;
          v42 = v9;
          v46 = v16;
          v30 = sub_AF5140(v20, 0);
          v16 = v46;
          v9 = v42;
          v15 = v38;
          v18 = v36;
          v17 = v34;
          if ( v39 == v30 )
          {
            result = 0;
            if ( v32 != (__int64 *)(*(_QWORD *)(v46 + 864) + 8LL * *(unsigned int *)(v46 + 880)) )
              result = v20;
            goto LABEL_11;
          }
        }
      }
LABEL_9:
      v21 = v17 + v18;
      ++v17;
      v18 = v15 & v21;
    }
  }
  result = 0;
LABEL_11:
  if ( v9 > 0x40 && v55 )
  {
    v49 = result;
    j_j___libc_free_0_0(v55);
    result = v49;
  }
  if ( v54 > 0x40 && v53 )
  {
    v50 = result;
    j_j___libc_free_0_0(v53);
    result = v50;
  }
  if ( !result && a6 )
  {
LABEL_19:
    v23 = *a1;
    v55 = a4;
    v24 = v23 + 856;
    v25 = sub_B97910(32, 1, a5);
    v27 = v25;
    if ( v25 )
    {
      v28 = a3;
      v52 = v25;
      sub_AF2C10(v25, v7, a5, a2, v28, v26, (__int64)&v55, 1);
      v27 = v52;
    }
    return sub_B044D0(v27, a5, v24);
  }
  return result;
}
