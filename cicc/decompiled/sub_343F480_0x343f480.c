// Function: sub_343F480
// Address: 0x343f480
//
__int64 __fastcall sub_343F480(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        char a8)
{
  __int64 result; // rax
  unsigned __int8 *v14; // rax
  __int64 v15; // r9
  unsigned int v16; // esi
  int v17; // edx
  _QWORD *v18; // rdi
  unsigned int *v19; // rax
  __int64 v20; // rdx
  int v21; // edx
  _QWORD *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdx
  int v26; // edx
  unsigned __int8 *v27; // rax
  int v28; // edx
  __int128 v29; // [rsp-20h] [rbp-A0h]
  __int128 v30; // [rsp-20h] [rbp-A0h]
  __int128 v31; // [rsp-20h] [rbp-A0h]
  __int128 v32; // [rsp-20h] [rbp-A0h]
  __int128 v33; // [rsp-10h] [rbp-90h]
  __int128 v34; // [rsp-10h] [rbp-90h]
  __int128 v35; // [rsp-10h] [rbp-90h]
  __int128 v36; // [rsp-10h] [rbp-90h]
  unsigned __int8 *v40; // [rsp+40h] [rbp-40h]

  if ( !a8 )
  {
    if ( !**(_BYTE **)(a1 + 8) )
    {
      result = **(unsigned __int8 **)(a1 + 48);
      if ( !(_BYTE)result )
        return result;
      *((_QWORD *)&v36 + 1) = a5;
      *(_QWORD *)&v36 = a4;
      *((_QWORD *)&v32 + 1) = a3;
      *(_QWORD *)&v32 = a2;
      v27 = sub_3406EB0(
              *(_QWORD **)(a1 + 16),
              0x3Au,
              *(_QWORD *)(a1 + 24),
              **(unsigned int **)(a1 + 56),
              *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL),
              a6,
              v32,
              v36);
      v15 = a6;
      v16 = 172;
      *(_QWORD *)a6 = v27;
      *(_DWORD *)(a6 + 8) = v28;
      v18 = *(_QWORD **)(a1 + 16);
      v19 = *(unsigned int **)(a1 + 56);
      v20 = *(_QWORD *)(a1 + 24);
LABEL_8:
      *((_QWORD *)&v34 + 1) = a5;
      *(_QWORD *)&v34 = a4;
      *((_QWORD *)&v30 + 1) = a3;
      *(_QWORD *)&v30 = a2;
      *(_QWORD *)a7 = sub_3406EB0(v18, v16, v20, *v19, *((_QWORD *)v19 + 1), v15, v30, v34);
      *(_DWORD *)(a7 + 8) = v21;
      return 1;
    }
    v22 = *(_QWORD **)(a1 + 16);
    v23 = *(_QWORD *)(a1 + 32);
    v24 = 64;
    v25 = *(_QWORD *)(a1 + 24);
LABEL_10:
    *((_QWORD *)&v35 + 1) = a5;
    *(_QWORD *)&v35 = a4;
    *((_QWORD *)&v31 + 1) = a3;
    *(_QWORD *)&v31 = a2;
    v40 = sub_3411F20(v22, v24, v25, *(unsigned int **)v23, *(_QWORD *)(v23 + 8), a6, v31, v35);
    *(_QWORD *)a6 = v40;
    *(_DWORD *)(a6 + 8) = v26;
    *(_QWORD *)a7 = v40;
    *(_DWORD *)(a7 + 8) = 1;
    return 1;
  }
  if ( **(_BYTE **)a1 )
  {
    v22 = *(_QWORD **)(a1 + 16);
    v23 = *(_QWORD *)(a1 + 32);
    v24 = 63;
    v25 = *(_QWORD *)(a1 + 24);
    goto LABEL_10;
  }
  result = **(unsigned __int8 **)(a1 + 40);
  if ( (_BYTE)result )
  {
    *((_QWORD *)&v33 + 1) = a5;
    *(_QWORD *)&v33 = a4;
    *((_QWORD *)&v29 + 1) = a3;
    *(_QWORD *)&v29 = a2;
    v14 = sub_3406EB0(
            *(_QWORD **)(a1 + 16),
            0x3Au,
            *(_QWORD *)(a1 + 24),
            **(unsigned int **)(a1 + 56),
            *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL),
            a6,
            v29,
            v33);
    v15 = a6;
    v16 = 173;
    *(_QWORD *)a6 = v14;
    *(_DWORD *)(a6 + 8) = v17;
    v18 = *(_QWORD **)(a1 + 16);
    v19 = *(unsigned int **)(a1 + 56);
    v20 = *(_QWORD *)(a1 + 24);
    goto LABEL_8;
  }
  return result;
}
