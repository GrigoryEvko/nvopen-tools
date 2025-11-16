// Function: sub_B05580
// Address: 0xb05580
//
__int64 __fastcall sub_B05580(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9,
        unsigned int a10,
        char a11)
{
  __int64 *v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r10
  int v15; // r15d
  __int64 *v16; // rbx
  __int64 v17; // r12
  int v18; // r15d
  __int64 v19; // rax
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  _BYTE *v22; // rax
  int v23; // r15d
  __int64 *v24; // r15
  __int64 result; // rax
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // r15
  int v30; // [rsp+Ch] [rbp-B4h]
  int v31; // [rsp+10h] [rbp-B0h]
  unsigned int v32; // [rsp+14h] [rbp-ACh]
  __int64 v33; // [rsp+20h] [rbp-A0h]
  __int64 v34; // [rsp+28h] [rbp-98h]
  __int64 v35; // [rsp+30h] [rbp-90h]
  __int64 v36; // [rsp+38h] [rbp-88h]
  __int64 v37; // [rsp+40h] [rbp-80h]
  __int128 v38; // [rsp+50h] [rbp-70h] BYREF
  __int64 v39; // [rsp+60h] [rbp-60h] BYREF
  __int64 v40; // [rsp+68h] [rbp-58h]
  __int64 v41; // [rsp+70h] [rbp-50h]
  __int64 v42; // [rsp+78h] [rbp-48h]
  int v43; // [rsp+80h] [rbp-40h]
  int v44[15]; // [rsp+84h] [rbp-3Ch] BYREF

  v12 = a1;
  v13 = a3;
  if ( a10 )
    goto LABEL_20;
  v14 = *a1;
  *((_QWORD *)&v38 + 1) = a3;
  v39 = a4;
  LODWORD(v38) = a2;
  v40 = a5;
  v42 = a7;
  v41 = a6;
  v43 = a8;
  v36 = v14;
  v44[0] = a9;
  v15 = *(_DWORD *)(v14 + 1552);
  v37 = *(_QWORD *)(v14 + 1536);
  if ( !v15 )
    goto LABEL_19;
  v33 = a6;
  v34 = a5;
  v31 = 1;
  v30 = v15 - 1;
  v32 = (v15 - 1) & sub_AFB0E0((int *)&v38, (__int64 *)&v38 + 1, &v39, v44);
  v35 = v13;
  while ( 1 )
  {
    v16 = (__int64 *)(v37 + 8LL * v32);
    v17 = *v16;
    if ( *v16 == -8192 )
      goto LABEL_8;
    if ( v17 == -4096 )
      goto LABEL_24;
    v18 = v38;
    if ( v18 == (unsigned __int16)sub_AF18C0(*v16) )
    {
      v19 = sub_AF5140(v17, 2u);
      if ( *((_QWORD *)&v38 + 1) == v19 )
      {
        v20 = sub_A17150((_BYTE *)(v17 - 16));
        if ( v39 == *((_QWORD *)v20 + 3) )
        {
          v21 = sub_A17150((_BYTE *)(v17 - 16));
          if ( v40 == *((_QWORD *)v21 + 4) )
          {
            v22 = sub_A17150((_BYTE *)(v17 - 16));
            if ( v41 == *((_QWORD *)v22 + 5) && v42 == *(_QWORD *)(v17 + 24) )
            {
              v23 = v43;
              if ( v23 == (unsigned int)sub_AF18D0(v17) && v44[0] == *(_DWORD *)(v17 + 44) )
                break;
            }
          }
        }
      }
    }
    v17 = *v16;
LABEL_8:
    if ( v17 == -4096 )
    {
LABEL_24:
      v12 = a1;
      v13 = v35;
      a5 = v34;
      a6 = v33;
      goto LABEL_19;
    }
    v32 = v30 & (v31 + v32);
    ++v31;
  }
  v24 = (__int64 *)(v37 + 8LL * v32);
  v12 = a1;
  v13 = v35;
  a5 = v34;
  a6 = v33;
  if ( v24 == (__int64 *)(*(_QWORD *)(v36 + 1536) + 8LL * *(unsigned int *)(v36 + 1552)) || (result = *v24) == 0 )
  {
LABEL_19:
    result = 0;
    if ( a11 )
    {
LABEL_20:
      v26 = *v12;
      v40 = a4;
      v39 = v13;
      v27 = v26 + 1528;
      v41 = a5;
      v42 = a6;
      v38 = 0;
      v28 = sub_B97910(48, 6, a10);
      v29 = v28;
      if ( v28 )
      {
        sub_B971C0(v28, (_DWORD)v12, 34, a10, (unsigned int)&v38, 6, 0, 0);
        *(_QWORD *)(v29 + 16) = 0;
        *(_QWORD *)(v29 + 32) = 0;
        *(_WORD *)(v29 + 2) = a2;
        *(_DWORD *)(v29 + 40) = 0;
        *(_QWORD *)(v29 + 24) = a7;
        *(_DWORD *)(v29 + 4) = a8;
        *(_DWORD *)(v29 + 44) = a9;
      }
      return sub_B054A0(v29, a10, v27);
    }
  }
  return result;
}
