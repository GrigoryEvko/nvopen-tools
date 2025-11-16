// Function: sub_76D880
// Address: 0x76d880
//
__int64 __fastcall sub_76D880(__int64 a1, int a2, __int64 *a3)
{
  __int64 *v4; // r14
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 *v7; // rdx

  v4 = (__int64 *)*a3;
  result = (__int64)qword_4D03E98;
  if ( qword_4D03E98 )
    qword_4D03E98 = (_QWORD *)*qword_4D03E98;
  else
    result = sub_823970(64);
  if ( !a2 )
  {
    if ( !v4 )
    {
      v7 = (__int64 *)qword_4F08050;
      if ( !qword_4F08050 )
      {
        qword_4F08050 = result;
        goto LABEL_8;
      }
      do
      {
        v4 = v7;
        v7 = (__int64 *)*v7;
      }
      while ( v7 );
    }
    *v4 = result;
LABEL_8:
    *(_QWORD *)result = 0;
    *a3 = result;
    goto LABEL_5;
  }
  v6 = qword_4F08050;
  qword_4F08050 = result;
  *(_QWORD *)result = v6;
LABEL_5:
  *(_QWORD *)(result + 8) = a1;
  *(_QWORD *)(a1 + 264) = result;
  *(_DWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)(result + 48) = 0;
  *(_DWORD *)(result + 56) = 0;
  return result;
}
