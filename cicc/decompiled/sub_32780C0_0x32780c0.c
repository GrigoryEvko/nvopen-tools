// Function: sub_32780C0
// Address: 0x32780c0
//
__int64 __fastcall sub_32780C0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        char a7,
        int a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        int a14,
        __int64 a15)
{
  unsigned __int16 *v16; // rdx
  __int64 v17; // r14
  __int64 v18; // r15
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rdx
  int v24; // r9d
  __int64 v25; // rdi
  __int128 *v26; // rax
  int v27; // ecx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int128 v31; // [rsp-20h] [rbp-B0h]
  __int128 v32; // [rsp-10h] [rbp-A0h]
  __int64 v33; // [rsp+0h] [rbp-90h]
  __int64 v35; // [rsp+18h] [rbp-78h]
  _QWORD v36[2]; // [rsp+20h] [rbp-70h] BYREF
  int v37; // [rsp+30h] [rbp-60h] BYREF
  __int64 v38; // [rsp+38h] [rbp-58h]
  __int16 v39; // [rsp+40h] [rbp-50h] BYREF
  __int64 v40; // [rsp+48h] [rbp-48h]
  __int64 v41; // [rsp+50h] [rbp-40h]
  __int64 v42; // [rsp+58h] [rbp-38h]

  v16 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v36[0] = a4;
  v17 = a9;
  v18 = *a1;
  v36[1] = a5;
  v35 = a10;
  v19 = *v16;
  v20 = *((_QWORD *)v16 + 1);
  LOWORD(v37) = v19;
  v38 = v20;
  if ( (_WORD)v19 )
  {
    if ( (unsigned __int16)(v19 - 17) > 0xD3u )
    {
      v39 = v19;
      v40 = v20;
      goto LABEL_4;
    }
    LOWORD(v19) = word_4456580[v19 - 1];
    v22 = 0;
  }
  else
  {
    v33 = v20;
    if ( !sub_30070B0((__int64)&v37) )
    {
      v40 = v33;
      v39 = 0;
      goto LABEL_9;
    }
    LOWORD(v19) = sub_3009970((__int64)&v37, a2, v33, v29, v30);
  }
  v39 = v19;
  v40 = v22;
  if ( !(_WORD)v19 )
  {
LABEL_9:
    v41 = sub_3007260((__int64)&v39);
    LODWORD(v21) = v41;
    v42 = v23;
    goto LABEL_10;
  }
LABEL_4:
  if ( (_WORD)v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
    BUG();
  v21 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v19 - 16];
LABEL_10:
  if ( !(unsigned __int8)sub_3277840(v17, v35, a11, a12, v21, v18, 1) )
    return 0;
  v25 = *a1;
  v26 = (__int128 *)v36;
  if ( !a6 )
    v26 = (__int128 *)&a7;
  v27 = a13;
  if ( !a6 )
    v27 = a14;
  v32 = *v26;
  *((_QWORD *)&v31 + 1) = a3;
  *(_QWORD *)&v31 = a2;
  a13 = v27;
  return sub_3406EB0(v25, v27, a15, v37, v38, v24, v31, v32);
}
