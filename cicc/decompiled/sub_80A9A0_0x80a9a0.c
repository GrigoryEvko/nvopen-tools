// Function: sub_80A9A0
// Address: 0x80a9a0
//
void __fastcall sub_80A9A0(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rax
  size_t v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // rdx

  if ( a1 )
  {
    v1 = a1;
    do
    {
      while ( 1 )
      {
        v2 = (_QWORD *)qword_4F18BA0;
        v3 = v1[5];
        if ( qword_4F18BA0 )
          break;
LABEL_11:
        v6 = sub_725260();
        *v6 = qword_4F18BA0;
        v7 = v1[5];
        qword_4F18BA0 = (__int64)v6;
        v6[1] = v7;
        v1 = (_QWORD *)*v1;
        if ( !v1 )
          return;
      }
      while ( 1 )
      {
        v4 = v2[1];
        if ( v3 == v4 )
          break;
        v5 = *(_QWORD *)(v3 + 176);
        if ( v5 == *(_QWORD *)(v4 + 176) && !memcmp(*(const void **)(v3 + 184), *(const void **)(v4 + 184), v5) )
          break;
        v2 = (_QWORD *)*v2;
        if ( !v2 )
          goto LABEL_11;
      }
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
  }
}
