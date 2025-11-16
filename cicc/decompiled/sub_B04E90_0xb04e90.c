// Function: sub_B04E90
// Address: 0xb04e90
//
__int64 __fastcall sub_B04E90(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        int a7,
        int a8,
        unsigned int a9,
        char a10)
{
  __int64 v10; // r11
  __int64 *v11; // r12
  int v12; // ebx
  __int64 v13; // r9
  int v14; // r15d
  unsigned int i; // r15d
  __int64 *v16; // rbx
  __int64 v17; // r12
  int v18; // ecx
  __int64 v19; // rax
  __int64 *v20; // r15
  __int64 result; // rax
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rdi
  int v26; // [rsp+Ch] [rbp-A4h]
  int v27; // [rsp+10h] [rbp-A0h]
  int v28; // [rsp+10h] [rbp-A0h]
  int v29; // [rsp+14h] [rbp-9Ch]
  __int64 v30; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+30h] [rbp-80h]
  __int64 v33; // [rsp+38h] [rbp-78h]
  _BYTE v37[24]; // [rsp+50h] [rbp-60h] BYREF
  int v38; // [rsp+68h] [rbp-48h] BYREF
  int v39; // [rsp+6Ch] [rbp-44h] BYREF
  int v40; // [rsp+70h] [rbp-40h]
  int v41; // [rsp+74h] [rbp-3Ch]

  v10 = a3;
  v11 = a1;
  v12 = a5;
  if ( a9 )
    goto LABEL_18;
  v13 = *a1;
  *(_DWORD *)v37 = a2;
  *(_QWORD *)&v37[8] = a3;
  *(_QWORD *)&v37[16] = a4;
  v38 = a5;
  v39 = a6;
  v40 = a7;
  v41 = a8;
  v14 = *(_DWORD *)(v13 + 912);
  v32 = *(_QWORD *)(v13 + 896);
  if ( !v14 )
    goto LABEL_17;
  v26 = v14 - 1;
  v29 = 1;
  v30 = v13;
  for ( i = (v14 - 1) & sub_AF9B00((int *)v37, (__int64 *)&v37[8], (__int64 *)&v37[16], &v38, &v39); ; i = v18 )
  {
    v16 = (__int64 *)(v32 + 8LL * i);
    v17 = *v16;
    if ( *v16 == -4096 )
    {
LABEL_22:
      v11 = a1;
      v10 = a3;
      v12 = a5;
      goto LABEL_17;
    }
    if ( v17 != -8192 )
      break;
LABEL_8:
    if ( v17 == -4096 )
      goto LABEL_22;
    v18 = v26 & (i + v29++);
  }
  v27 = *(_DWORD *)v37;
  if ( v27 != (unsigned __int16)sub_AF18C0(*v16)
    || (v19 = sub_AF5140(v17, 2u), *(_OWORD *)&v37[8] != __PAIR128__(*(_QWORD *)(v17 + 24), v19))
    || (v28 = v38, v28 != (unsigned int)sub_AF18D0(v17))
    || v39 != *(_DWORD *)(v17 + 44)
    || v40 != *(_DWORD *)(v17 + 40)
    || v41 != *(_DWORD *)(v17 + 20) )
  {
    v17 = *v16;
    goto LABEL_8;
  }
  v20 = (__int64 *)(v32 + 8LL * i);
  v11 = a1;
  v10 = a3;
  v12 = a5;
  if ( v20 == (__int64 *)(*(_QWORD *)(v30 + 896) + 8LL * *(unsigned int *)(v30 + 912)) || (result = *v20) == 0 )
  {
LABEL_17:
    result = 0;
    if ( a10 )
    {
LABEL_18:
      v22 = *v11;
      *(_QWORD *)&v37[16] = v10;
      *(_OWORD *)v37 = 0;
      v23 = v22 + 888;
      v24 = sub_B97910(48, 3, a9);
      v25 = v24;
      if ( v24 )
      {
        v33 = v24;
        sub_B971C0(v24, (_DWORD)v11, 12, a9, (unsigned int)v37, 3, 0, 0);
        v25 = v33;
        *(_DWORD *)(v33 + 20) = a8;
        *(_WORD *)(v33 + 2) = a2;
        *(_QWORD *)(v33 + 24) = a4;
        *(_DWORD *)(v33 + 16) = 0;
        *(_DWORD *)(v33 + 40) = a7;
        *(_DWORD *)(v33 + 4) = v12;
        *(_QWORD *)(v33 + 32) = 0;
        *(_DWORD *)(v33 + 44) = a6;
      }
      return sub_B04CA0(v25, a9, v23);
    }
  }
  return result;
}
